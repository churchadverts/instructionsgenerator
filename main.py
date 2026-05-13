import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
openai_key   = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

client_ai : AsyncOpenAI = AsyncOpenAI(api_key=openai_key)
supabase  : Client       = create_client(supabase_url, supabase_key)

OPENAI_MODEL        = "gpt-4.1-mini"
HISTORY_DAYS        = int(os.getenv("HISTORY_DAYS", "90"))
MIN_OWNER_MSGS      = 10    # below this: voice analysis uses low-confidence fallback
MIN_CUSTOMER_CONVS  = 5     # below this: pattern analysis uses minimal fallback
MAX_OWNER_MSGS      = 200   # hard cap before time-based sampling kicks in
MAX_CONVERSATIONS   = 30    # max conversations to pull for pattern analysis
MAX_MSGS_PER_CONV   = 15    # messages shown per conversation thread
MAX_CONVS_PER_OUTCOME = 8   # max conversations per outcome label

# ==========================================
# 2. SUPABASE HELPERS
# ==========================================

def update_persona_status(business_id: str, status: str):
    try:
        payload = {"persona_pack_status": status}
        if status in ("ready", "failed"):
            payload["persona_pack_last_run_at"] = datetime.now(timezone.utc).isoformat()
        supabase.table("businesses") \
            .update(payload) \
            .eq("business_id", business_id) \
            .execute()
        print(f"  [DB] persona_pack_status → {status}")
    except Exception as e:
        print(f"  [DB] Failed to update persona status: {e}")


def fetch_bot_config(bot_id: str) -> dict:
    try:
        res = (
            supabase.table("ai_bots_config")
            .select("prompt, model, temperature, max_tokens")
            .eq("bot_id", bot_id)
            .eq("is_active", True)
            .single()
            .execute()
        )
        if res.data:
            return res.data
    except Exception:
        pass
    print(f"  [Config] '{bot_id}' not in ai_bots_config — using fallback")
    return {}


def save_persona_pack(business_id: str, pack: dict) -> bool:
    try:
        # Deactivate any existing active pack for this business
        supabase.table("persona_packs") \
            .update({"is_active": False}) \
            .eq("business_id", business_id) \
            .eq("is_active", True) \
            .execute()

        # Get next version number
        res = supabase.table("persona_packs") \
            .select("version") \
            .eq("business_id", business_id) \
            .order("version", desc=True) \
            .limit(1) \
            .execute()
        version = (res.data[0]["version"] + 1) if res.data else 1

        supabase.table("persona_packs").insert({
            "business_id":  business_id,
            "version":      version,
            "pack":         pack,
            "is_active":    True,
            "generated_by": "system"
        }).execute()

        print(f"  [DB] Persona pack v{version} saved")
        return True
    except Exception as e:
        print(f"  [DB] Failed to save persona pack: {e}")
        return False

# ==========================================
# 3. DATA PREPARATION
# ==========================================

def get_outcome_label(conv: dict) -> str:
    """Maps lead_state + lead_quality to a clean outcome label for the AI."""
    state   = conv.get("lead_state", "")
    quality = conv.get("lead_quality", "")

    if state == "won":                                              return "WON"
    if state in ("lost", "do_not_contact"):                        return "LOST"
    if state == "ghosted":                                         return "GHOSTED"
    if state in ("engaged", "warm") and quality == "hot":          return "WON"
    if state == "stalled" and quality == "cold":                   return "LOST"
    return "IN_PROGRESS"


def fetch_business_data(business_id: str) -> dict:
    """
    Reads and shapes all raw data into three buckets:
    - website_text:            combined scraped pages
    - owner_messages:          sampled outbound admin messages (voice source)
    - customer_conversations:  inbound threads grouped by outcome
    - business_info:           basic details from businesses table
    """
    print(f"  [Data] Fetching data for {business_id}...")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).isoformat()

    # ── Business info ────────────────────────────────────────────
    biz_res = supabase.table("businesses") \
        .select("business_id, name, industry, business_type, currency, timezone, website_url") \
        .eq("business_id", business_id) \
        .single() \
        .execute()
    business_info = biz_res.data or {}

    # ── Website data ─────────────────────────────────────────────
    pages_res = supabase.table("raw_website_data") \
        .select("page_type, raw_text, scraped_at") \
        .eq("business_id", business_id) \
        .order("scraped_at", desc=True) \
        .execute()

    # One section per page type — most recently scraped version only
    seen_types      = set()
    website_sections = []
    for page in (pages_res.data or []):
        if page["page_type"] not in seen_types:
            seen_types.add(page["page_type"])
            website_sections.append(
                f"[{page['page_type'].upper()} PAGE]\n{page['raw_text']}"
            )
    website_text = "\n\n".join(website_sections)

    # ── Owner messages (voice source) ────────────────────────────
    owner_res = supabase.table("messages") \
        .select("content, type, created_at") \
        .eq("business_id", business_id) \
        .eq("direction", "out") \
        .eq("role", "admin") \
        .gte("created_at", cutoff) \
        .order("created_at", desc=False) \
        .limit(MAX_OWNER_MSGS) \
        .execute()

    # Text messages only — voice analysis needs words, not [image] placeholders
    owner_messages = [
        m["content"].get("text", "").strip()
        for m in (owner_res.data or [])
        if m.get("type") == "text"
        and m.get("content", {}).get("text", "").strip()
    ]

    # Sample evenly across time when we have many messages
    # This prevents the profile from skewing toward recent style only
    if len(owner_messages) > 100:
        step           = len(owner_messages) // 100
        owner_messages = owner_messages[::step][:100]

    # ── Customer conversations (pattern source) ──────────────────
    convs_res = supabase.table("conversations") \
        .select("id, lead_state, lead_quality, conv_stage, customer_intent") \
        .eq("business_id", business_id) \
        .neq("lead_state", "new") \
        .limit(MAX_CONVERSATIONS) \
        .execute()

    conversations = convs_res.data or []
    conv_ids      = [c["id"] for c in conversations]
    msgs_by_conv  = defaultdict(list)

    if conv_ids:
        msgs_res = supabase.table("messages") \
            .select("conversation_id, direction, content, type, created_at") \
            .in_("conversation_id", conv_ids) \
            .order("created_at", desc=False) \
            .execute()

        for msg in (msgs_res.data or []):
            text     = msg.get("content", {}).get("text", "")
            msg_type = msg.get("type", "text")
            # Include media types as a signal — even without text
            display  = text if text else f"[{msg_type}]"
            msgs_by_conv[msg["conversation_id"]].append({
                "direction": msg["direction"],
                "text":      display,
                "type":      msg_type
            })

    # Build labeled conversation threads
    customer_conversations = []
    for conv in conversations:
        thread_msgs = msgs_by_conv.get(conv["id"], [])
        if not thread_msgs:
            continue
        outcome = get_outcome_label(conv)
        lines   = []
        for msg in thread_msgs[:MAX_MSGS_PER_CONV]:
            speaker = "Business" if msg["direction"] == "out" else "Customer"
            lines.append(f"{speaker}: {msg['text']}")
        customer_conversations.append({
            "outcome": outcome,
            "thread":  "\n".join(lines),
            "intent":  conv.get("customer_intent") or "",
            "stage":   conv.get("conv_stage") or ""
        })

    print(
        f"  [Data] pages:{len(seen_types)} | "
        f"owner_msgs:{len(owner_messages)} | "
        f"customer_convs:{len(customer_conversations)}"
    )

    return {
        "business_info":            business_info,
        "website_text":             website_text,
        "owner_messages":           owner_messages,
        "customer_conversations":   customer_conversations,
        "has_enough_voice_data":    len(owner_messages) >= MIN_OWNER_MSGS,
        "has_enough_customer_data": len(customer_conversations) >= MIN_CUSTOMER_CONVS
    }

# ==========================================
# 4. FALLBACK PROMPTS
# ==========================================

VOICE_FALLBACK = """
You are a linguistic analyst specialising in WhatsApp business communication in African markets.
Analyse the collection of WhatsApp messages written by a business owner and extract their
communication profile. Return ONLY a valid JSON object. No preamble. No markdown fences.

Extract these fields:
formality_score (int 1-10), language_mix (object: english/swahili/sheng as decimal percentages),
greeting_style (string), closing_style (string),
emoji_usage (object: frequency/typical_emojis/context),
sentence_structure (string), punctuation_habits (array of strings),
signature_phrases (array up to 8), tone_descriptors (array 3-5 adjectives),
response_length_pattern (short|medium|long|mixed),
urgency_language (array), trust_building_language (array),
observations (string). Use null for fields with no evidence.
"""

CONTEXT_FALLBACK = """
You are a business analyst specialising in African SME markets.
Analyse the website content and conversation sample to extract a structured business profile.
Return ONLY a valid JSON object. No preamble. No markdown fences.

Extract these fields:
core_offer (string, 2 sentences max), target_customer (string),
key_services (array of objects: name/description/price_indicator),
unique_selling_points (array), top_objections (array of objects: objection/how_business_handles_it/effectiveness),
closing_triggers (array), payment_methods (array — always check for M-Pesa),
delivery_or_fulfillment (string), price_sensitivity_signals (string),
social_proof_used (array), knowledge_gaps (array), observations (string).
Do not invent data. Use null for missing fields.
"""

CUSTOMER_FALLBACK = """
You are a behavioural analyst specialising in WhatsApp customer communication patterns in Kenya.
Analyse the labeled conversation threads and extract buyer behaviour patterns.
Return ONLY a valid JSON object. No preamble. No markdown fences.

Extract these fields:
buyer_profiles (array of objects: profile_name/description/trigger_phrases/behaviour_pattern/
dominant_concern/best_approach/what_kills_the_deal/typical_outcome_distribution),
ghost_patterns (object: last_message_types/common_ghost_triggers/reengagement_signals),
loss_patterns (object: common_stated_reasons/common_unstated_reasons/point_of_failure),
win_patterns (object: customer_signals_before_close/conversation_arc/average_touchpoints/decision_speed),
language_observations (object: language_mix/formality_level/emotional_expressiveness/cultural_signals),
sentiment_vocabulary (object: positive_signals/hesitation_signals/negative_signals/
urgency_signals/price_resistance_signals/trust_deficit_signals as arrays),
observations (string).
"""

PACK_FALLBACK = """
You are a senior AI systems architect building a WhatsApp follow-up agent persona.
Synthesise the three analysis reports into one comprehensive Persona Pack JSON.
Return ONLY a valid JSON object. No preamble. No markdown fences.

The pack must include all of these top-level fields:
business_id, business_name, generated_at, version,
persona (display_name/voice_tone/formality_score/language_mix/emoji_style/
typical_greeting/typical_closing/signature_phrases/phrases_to_avoid/
tone_descriptors/sentence_length/confidence),
business_context (core_offer/target_customer/key_services/unique_selling_points/
payment_methods/delivery_info/social_proof_available/knowledge_gaps/confidence),
objection_playbook (array: objection/response_strategy/suggested_language/escalation_if_repeated),
closing_triggers (array),
customer_profiles (array: profile_id/profile_name/detection_signals/approach_strategy/
message_style_adjustment/cta_style/what_to_avoid),
follow_up_rules (max_follow_ups_before_exit/follow_up_intervals_hours/quiet_hours_start/
quiet_hours_end/active_days/ghost_wait_hours/escalation_path/exit_message_strategy),
sentiment_response_map (positive/neutral/hesitant/price_resistant/time_poor/
trust_deficit/negative/aggressive),
winning_conversation_arc (array of steps),
reengagement_strategy (when_to_use/angle/tone_shift/max_reengagement_attempts),
human_handoff_triggers (array),
pack_confidence_overall (low|medium|high),
pack_notes (string).
"""

# ==========================================
# 5. GENERIC AI CALLER
# ==========================================

async def run_ai_call(
    bot_id:          str,
    fallback_prompt: str,
    user_content:    str,
    label:           str
) -> dict | None:
    """
    Fetches bot config from ai_bots_config, calls OpenAI, parses and returns JSON.
    Uses response_format json_object to guarantee clean output.
    """
    config        = fetch_bot_config(bot_id)
    system_prompt = config.get("prompt",      fallback_prompt)
    model         = config.get("model",       OPENAI_MODEL)
    max_tokens    = config.get("max_tokens",  3000)
    temperature   = config.get("temperature", 0.2)

    try:
        response = await client_ai.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content}
            ]
        )
        result = json.loads(response.choices[0].message.content.strip())
        print(f"  [{label}] ✓ Done")
        return result
    except Exception as e:
        print(f"  [{label}] ✗ {e}")
        return None

# ==========================================
# 6. THE THREE ANALYSERS
# ==========================================

async def run_voice_analyser(data: dict) -> dict:
    if not data["has_enough_voice_data"]:
        count = len(data["owner_messages"])
        print(f"  [Voice] Only {count} owner messages — returning low-confidence fallback")
        return {
            "formality_score":    5,
            "language_mix":       {"english": 0.7, "swahili": 0.3, "sheng": 0.0},
            "tone_descriptors":   ["professional", "helpful"],
            "signature_phrases":  [],
            "observations":       (
                f"Insufficient owner messages for reliable voice analysis. "
                f"{count} found, {MIN_OWNER_MSGS} required. "
                f"Profile will improve as more conversations accumulate."
            ),
            "_confidence": "low",
            "_source":     "fallback"
        }

    numbered = "\n".join(
        f"{i+1}. {msg}"
        for i, msg in enumerate(data["owner_messages"])
    )

    user_content = (
        f"Business Name: {data['business_info'].get('name', 'Unknown')}\n"
        f"Country: Kenya\n\n"
        f"Owner WhatsApp Messages to Analyse:\n{numbered}"
    )

    result = await run_ai_call(
        "voice_tone_analyser", VOICE_FALLBACK, user_content, "Voice"
    )
    return result or {
        "observations": "Voice analysis failed",
        "_confidence": "low"
    }


async def run_context_extractor(data: dict) -> dict | None:
    # Use first 3 conversations as a real-chat sample for the context bot
    sample_threads = data["customer_conversations"][:3]
    chat_sample    = "\n\n".join(
        f"[Conversation — Outcome: {c['outcome']}]\n{c['thread']}"
        for c in sample_threads
    ) or "No conversation history available yet."

    user_content = (
        f"Business Name: {data['business_info'].get('name', 'Unknown')}\n\n"
        f"Website Content:\n{data['website_text'][:30000]}\n\n"
        f"Sample Chat History:\n{chat_sample}"
    )

    return await run_ai_call(
        "business_context_extractor", CONTEXT_FALLBACK, user_content, "Context"
    )


async def run_customer_analyser(data: dict) -> dict:
    if not data["has_enough_customer_data"]:
        count = len(data["customer_conversations"])
        print(f"  [Customer] Only {count} labeled conversations — returning minimal profile")
        return {
            "buyer_profiles": [],
            "ghost_patterns": {},
            "win_patterns":   {},
            "loss_patterns":  {},
            "observations":   (
                f"Insufficient conversation history for pattern analysis. "
                f"{count} labeled conversations found, {MIN_CUSTOMER_CONVS} required. "
                f"Patterns will improve as more conversations reach a terminal state."
            ),
            "_confidence": "low"
        }

    # Group threads by outcome, cap each group to avoid token imbalance
    grouped = defaultdict(list)
    for conv in data["customer_conversations"]:
        grouped[conv["outcome"]].append(conv["thread"])

    formatted = []
    for outcome, threads in grouped.items():
        for thread in threads[:MAX_CONVS_PER_OUTCOME]:
            formatted.append(
                f"[CONVERSATION {len(formatted)+1} — OUTCOME: {outcome}]\n{thread}"
            )

    outcome_distribution = {
        outcome: len(threads)
        for outcome, threads in grouped.items()
    }

    user_content = (
        f"Business Name: {data['business_info'].get('name', 'Unknown')}\n"
        f"Country: Kenya\n\n"
        f"Outcome Distribution: {json.dumps(outcome_distribution)}\n\n"
        f"Labeled Conversation Threads:\n\n" +
        "\n\n".join(formatted)
    )

    result = await run_ai_call(
        "customer_pattern_analyser", CUSTOMER_FALLBACK, user_content, "Customer"
    )
    return result or {
        "observations": "Customer pattern analysis failed",
        "_confidence": "low"
    }

# ==========================================
# 7. PERSONA PACK GENERATOR
# ==========================================

async def run_persona_pack_generator(
    voice_result:    dict,
    context_result:  dict,
    customer_result: dict,
    business_info:   dict,
    business_id:     str
) -> dict | None:
    print("  [Pack] Generating Persona Pack...")

    config        = fetch_bot_config("persona_pack_generator")
    system_prompt = config.get("prompt",      PACK_FALLBACK)
    model         = config.get("model",       OPENAI_MODEL)
    max_tokens    = config.get("max_tokens",  4096)
    temperature   = config.get("temperature", 0.3)

    user_content = (
        f"Business ID: {business_id}\n"
        f"Business Name: {business_info.get('name', 'Unknown')}\n"
        f"Country: Kenya\n"
        f"Industry: {business_info.get('industry', 'General')}\n"
        f"Currency: {business_info.get('currency', 'KES')}\n\n"
        f"INPUT REPORT 1 — Voice & Tone Analysis:\n"
        f"{json.dumps(voice_result, ensure_ascii=False)}\n\n"
        f"INPUT REPORT 2 — Business Context:\n"
        f"{json.dumps(context_result, ensure_ascii=False)}\n\n"
        f"INPUT REPORT 3 — Customer Patterns:\n"
        f"{json.dumps(customer_result, ensure_ascii=False)}"
    )

    try:
        response = await client_ai.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content}
            ]
        )
        pack = json.loads(response.choices[0].message.content.strip())

        # Ensure identity fields are always present
        pack["business_id"]   = business_id
        pack["business_name"] = business_info.get("name", "Unknown")
        pack["generated_at"]  = datetime.now(timezone.utc).isoformat()

        print("  [Pack] ✓ Persona Pack generated")
        return pack
    except Exception as e:
        print(f"  [Pack] ✗ {e}")
        return None

# ==========================================
# 8. MAIN PIPELINE
# ==========================================

async def generate_persona_pipeline(business_id: str):
    """
    Full System 1 pipeline:
    1. Guard against concurrent runs
    2. Fetch and prepare all data
    3. Run 3 analysers in parallel
    4. Generate Persona Pack from all 3 outputs
    5. Save versioned pack, update status
    """
    print(f"\n[System1] Pipeline starting — business: {business_id}")

    # ── Concurrent run guard ─────────────────────────────────────
    try:
        check = supabase.table("businesses") \
            .select("persona_pack_status") \
            .eq("business_id", business_id) \
            .single() \
            .execute()
        if check.data.get("persona_pack_status") == "running":
            print(f"[System1] Already running for {business_id} — skipping duplicate trigger")
            return
    except Exception as e:
        print(f"[System1] Status check failed: {e} — proceeding anyway")

    update_persona_status(business_id, "running")

    try:
        # ── Step 1: Prepare data ─────────────────────────────────
        data = fetch_business_data(business_id)

        # ── Step 2: 3 analysers in parallel ──────────────────────
        print("  [Pipeline] Running 3 analysers in parallel...")
        voice_result, context_result, customer_result = await asyncio.gather(
            run_voice_analyser(data),
            run_context_extractor(data),
            run_customer_analyser(data)
        )

        # Context extraction is the only fatal failure
        # Voice and customer return graceful fallbacks so they are never None here
        if context_result is None:
            raise Exception(
                "Business context extraction failed — cannot generate pack without it. "
                "Check that raw_website_data has content for this business."
            )

        # ── Step 3: Generate Persona Pack ────────────────────────
        pack = await run_persona_pack_generator(
            voice_result,
            context_result,
            customer_result,
            data["business_info"],
            business_id
        )

        if not pack:
            raise Exception("Persona Pack generator returned no result")

        # ── Step 4: Save ─────────────────────────────────────────
        if not save_persona_pack(business_id, pack):
            raise Exception("Database save failed for Persona Pack")

        update_persona_status(business_id, "ready")
        print(f"[System1] ✓ Pipeline complete — business: {business_id}\n")

    except Exception as e:
        print(f"[System1] ✗ Pipeline failed for {business_id}: {e}")
        update_persona_status(business_id, "failed")

# ==========================================
# 9. FASTAPI APP
# ==========================================

app = FastAPI(title="System 1 — Strategist AI")


class PersonaRequest(BaseModel):
    business_id: str


@app.post("/generate-persona")
async def generate_persona(req: PersonaRequest, background_tasks: BackgroundTasks):
    """
    Trigger persona pack generation. Returns immediately.
    Pipeline runs in the background.
    Called by: Evolution cleaner after history sync, orchestrator after scraping.
    """
    business_id = req.business_id.strip()
    if not business_id:
        raise HTTPException(status_code=400, detail="business_id is required")

    background_tasks.add_task(generate_persona_pipeline, business_id)

    return {
        "status":      "queued",
        "business_id": business_id,
        "message":     "Persona pack generation started in background"
    }


@app.get("/persona-status/{business_id}")
def get_persona_status(business_id: str):
    """
    Returns current status and the active pack when ready.
    Polled by the frontend every few seconds during the building screen.
    """
    try:
        biz = supabase.table("businesses") \
            .select("name, persona_pack_status, persona_pack_last_run_at") \
            .eq("business_id", business_id) \
            .single() \
            .execute()

        status = biz.data or {}
        pack   = None

        if status.get("persona_pack_status") == "ready":
            pack_res = supabase.table("persona_packs") \
                .select("pack, version, created_at") \
                .eq("business_id", business_id) \
                .eq("is_active", True) \
                .single() \
                .execute()
            pack = pack_res.data

        return {
            "business_id":              business_id,
            "persona_pack_status":      status.get("persona_pack_status"),
            "persona_pack_last_run_at": status.get("persona_pack_last_run_at"),
            "ready":                    status.get("persona_pack_status") == "ready",
            "pack":                     pack
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/refresh-persona/{business_id}")
async def refresh_persona(business_id: str, background_tasks: BackgroundTasks):
    """
    Force-regenerates the pack regardless of current status.
    Resets status to pending first so the concurrent run guard doesn't block it.
    Used for: dashboard refresh button, monthly cron job in Phase 7.
    """
    try:
        supabase.table("businesses") \
            .update({"persona_pack_status": "pending"}) \
            .eq("business_id", business_id) \
            .execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not reset status: {e}")

    background_tasks.add_task(generate_persona_pipeline, business_id)

    return {
        "status":      "queued",
        "business_id": business_id,
        "message":     "Persona pack refresh started"
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "system1-strategist"}


@app.get("/")
def read_root():
    return {"message": "System 1 Strategist AI active"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
