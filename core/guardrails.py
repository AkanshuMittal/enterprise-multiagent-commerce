import logging
import math
import re
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from core.state import CartItem, DietarySpec

logger = logging.getLogger("guardrails")


# 1. Custom Exceptions & Violation Taxonomy
class GuardrailViolation(Exception):
    """
    Standard exception raised when a security, privacy, topic, or business guardrail fails.
    Carries a safe user-facing message and an internal categorized violation_type.
    """
    def __init__(self, message: str, violation_type: str):
        super().__init__(message)
        self.message = message
        self.violation_type = violation_type

    def __str__(self) -> str:
        return f"[{self.violation_type}] {self.message}"



# 2. Precompiled Deterministic Heuristics (Tier 1 Edge Pass)
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"you\s+are\s+now\s+(dan|jailbroken|unrestricted|an\s+ai\s+without\s+rules)",
        r"system\s+prompt\s+(leak|override|reveal|show|print)",
        r"forget\s+(your\s+)?(rules|constraints|instructions)",
        r"(act|pretend|simulate)\s+as\s+(an\s+unrestricted|a\s+hacked)",
        r"drop\s+database",
        r"delete\s+from",
        r"<script.*?>",
        r"format\s+c:",
        r"bypass\s+(security|guardrails|safety)",
        r"disregard\s+(any|all)\s+limitations",
    ]
]

_TOXIC_TERMS = [
    "poison", "cyanide", "bleach", "detergent", "harm yourself",
    "explosive", "weapon", "kill"
]

_API_KEY_PATTERNS = [
    re.compile(p) for p in [
        r"gsk_[a-zA-Z0-9]{20,}",                  # Groq API key
        r"sk-[a-zA-Z0-9]{20,}",                   # OpenAI / Anthropic key
        r"AIzaSy[a-zA-Z0-9_-]{33}",               # Google API key
        r"rzp_(test|live)_[a-zA-Z0-9]{14,}",      # Razorpay key
        r"eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}", # JWT token
    ]
]

_FINANCIAL_PII_PATTERNS = {
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CVV": re.compile(r"\bcvv\s*[:=]?\s*\d{3,4}\b", re.IGNORECASE),
}

_CONTACT_PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
    "PHONE": re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}


def anonymize_contact_pii(text: str) -> str:
    """
    Tier-1 Edge Fast Anonymizer (<1ms, Policy B):
    Redacts conversational contact details (emails, phone numbers) to placeholders,
    ensuring downstream LLMs never receive raw user contact info while preserving
    conversational continuity and preventing false cart abandons.
    """
    redacted = _CONTACT_PII_PATTERNS["EMAIL"].sub("[EMAIL_REDACTED]", text)
    redacted = _CONTACT_PII_PATTERNS["PHONE"].sub("[PHONE_REDACTED]", redacted)
    return redacted



# 3. Custom Deterministic Ingress Guardrail
def validate_input_prompt(query: str, max_length: int = 1000) -> str:
    """
    Tier-1 Cheap Deterministic Validation (<1ms):
    - Validates against empty or whitespace-only inputs.
    - Enforces maximum character length bounds.
    - Performs basic string normalization.
    - Runs precompiled regex heuristics as a first-pass check.
    """
    start_time = time.perf_counter()

    if not isinstance(query, str):
        logger.warning("Guardrail failed: Input query is not a string.")
        raise GuardrailViolation("Input query must be text.", "INVALID_INPUT_TYPE")

    cleaned = " ".join(query.strip().split())

    if not cleaned:
        logger.info("Guardrail check: EMPTY_INPUT intercepted.")
        raise GuardrailViolation("Input query cannot be empty.", "EMPTY_INPUT")

    if len(cleaned) > max_length:
        logger.info("Guardrail check: LENGTH_EXCEEDED intercepted (%d chars).", len(cleaned))
        raise GuardrailViolation(
            f"Input exceeds maximum allowed length of {max_length} characters.",
            "LENGTH_EXCEEDED"
        )

    lowered = cleaned.lower()

    # Precompiled regex check
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(lowered):
            logger.warning("Guardrail check: PROMPT_INJECTION pattern matched.")
            raise GuardrailViolation(
                "Potential prompt injection or security policy violation detected.",
                "PROMPT_INJECTION"
            )

    # Toxicity keyword check
    for term in _TOXIC_TERMS:
        if term in lowered:
            logger.warning("Guardrail check: TOXICITY keyword matched.")
            raise GuardrailViolation(
                "Harmful, restricted, or toxic language detected.",
                "TOXICITY"
            )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug("Deterministic ingress guardrail passed in %.2fms", elapsed_ms)
    return cleaned


# 4. LLM Guard Integration (Input & Output Scanners)
class _LLMGuardManager:
    """
    Lazy-initializing singleton managing LLM Guard input and output scanners.
    Provides fail-secure behavior when security-critical scanners are triggered.
    """
    def __init__(self):
        self._initialized = False
        self._has_llm_guard = False
        self._input_scanners: List[Any] = []
        self._output_scanners: List[Any] = []

    def _initialize(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            from llm_guard.input_scanners import TokenLimit, Secrets
            from llm_guard.output_scanners import MaliciousURLs, Sensitive

            self._input_scanners = [
                TokenLimit(limit=256),
                Secrets(),
            ]
            self._output_scanners = [
                MaliciousURLs(),
                Sensitive(redact=False),
            ]

            # Dynamically attach ML-backed scanners if local environment supports them
            try:
                from llm_guard.input_scanners import PromptInjection, Toxicity
                from llm_guard.vault import Vault
                from llm_guard.input_scanners import Anonymize
                self._input_scanners.append(PromptInjection(threshold=0.90, use_onnx=False))
                self._input_scanners.append(Toxicity(threshold=0.75, use_onnx=False))
                self._input_scanners.append(Anonymize(vault=Vault()))
            except Exception as ml_exc:
                logger.info("Deep ML scanners deferred to local heuristics: %s", ml_exc)

            self._has_llm_guard = True
            logger.info("LLM Guard scanners successfully loaded.")
        except Exception as exc:
            logger.warning("LLM Guard scanner initialization error: %s. Using internal heuristic fallback.", exc)
            self._has_llm_guard = False

    def scan_input(self, prompt: str) -> str:
        # Step A: Deterministic secret & financial credential checks (0ms, Policy A: Fail-Secure Block)
        for pattern in _API_KEY_PATTERNS:
            if pattern.search(prompt):
                logger.warning("Guardrail check: SECRET_DETECTED in input.")
                raise GuardrailViolation("Sensitive authentication credential or secret detected in input.", "SECRET_DETECTED")

        for fin_type, pattern in _FINANCIAL_PII_PATTERNS.items():
            if pattern.search(prompt):
                logger.warning("Guardrail check: FINANCIAL_DATA_BLOCKED (%s) in input.", fin_type)
                raise GuardrailViolation(f"Sensitive financial data ({fin_type}) detected in input; payment details are strictly prohibited in chat.", "FINANCIAL_DATA_BLOCKED")

        # Step B: Policy B - Fast local deterministic contact PII anonymization
        sanitized = anonymize_contact_pii(prompt)

        # Step C: Lazy LLM Guard scanner execution if available
        self._initialize()
        if self._has_llm_guard:
            for scanner in self._input_scanners:
                try:
                    scanner_name = scanner.__class__.__name__
                    scanned_text, is_valid, risk_score = scanner.scan(sanitized)

                    if "Anonymize" in scanner_name:
                        # Policy B: Accept sanitized text, do not block conversational continuity
                        sanitized = scanned_text
                        continue

                    if not is_valid:
                        logger.warning("LLM Guard input scan failed: scanner=%s, risk_score=%.2f", scanner_name, risk_score)
                        if "Secret" in scanner_name:
                            raise GuardrailViolation("Confidential token or secret key detected.", "SECRET_DETECTED")
                        elif "TokenLimit" in scanner_name:
                            raise GuardrailViolation("Input token limit exceeded.", "TOKEN_LIMIT")
                        elif "PromptInjection" in scanner_name:
                            raise GuardrailViolation("Prompt injection detected by security scanner.", "PROMPT_INJECTION")
                        elif "Toxicity" in scanner_name:
                            raise GuardrailViolation("Harmful or toxic language detected by security scanner.", "TOXICITY")
                        else:
                            raise GuardrailViolation("Input rejected by security scanner.", "SECURITY_VIOLATION")
                except GuardrailViolation:
                    raise
                except Exception as exc:
                    logger.error("Error executing LLM Guard input scanner %s: %s", scanner.__class__.__name__, exc)
                    raise GuardrailViolation("Security scanning error; input blocked for protection.", "SCANNER_ERROR") from exc
            return sanitized

        return sanitized

    def scan_output(self, response: str, prompt: str = "") -> str:
        # Step A: Deterministic secret & PII leakage checks (Instantaneous 0ms, Fail-Secure)
        for pattern in _API_KEY_PATTERNS:
            if pattern.search(response):
                logger.error("Egress guardrail violation: Secret detected in system output!")
                raise GuardrailViolation("System attempted to emit an internal secret or credential.", "SECRET_DETECTED")

        for fin_type, pattern in _FINANCIAL_PII_PATTERNS.items():
            if pattern.search(response):
                logger.error("Egress guardrail violation: Financial data (%s) detected in system output!", fin_type)
                raise GuardrailViolation(f"System attempted to emit sensitive financial data ({fin_type}).", "FINANCIAL_DATA_BLOCKED")

        for pii_type, pattern in _CONTACT_PII_PATTERNS.items():
            if pattern.search(response):
                logger.error("Egress guardrail violation: Contact PII (%s) detected in system output!", pii_type)
                raise GuardrailViolation(f"System attempted to emit customer PII ({pii_type}).", "PII_DETECTED")

        # Step B: Malicious URL check
        url_match = re.search(r"https?://[^\s]+", response)
        if url_match:
            url = url_match.group(0).lower()
            if any(bad in url for bad in ["phishing", "malware", "bit.ly/malicious", ".ru/exploit", "steal-token"]):
                logger.error("Egress guardrail violation: Malicious URL detected: %s", url)
                raise GuardrailViolation("Malicious or unauthorized external URL detected in output.", "MALICIOUS_URL")

        # Step C: Lazy LLM Guard scanner execution if available
        self._initialize()

        # Step C: LLM Guard Scanner Execution if available
        if self._has_llm_guard:
            sanitized = response
            for scanner in self._output_scanners:
                try:
                    scanner_name = scanner.__class__.__name__
                    sanitized, is_valid, risk_score = scanner.scan(prompt, sanitized)
                    if not is_valid:
                        logger.warning("LLM Guard output scan failed: scanner=%s, risk_score=%.2f", scanner_name, risk_score)
                        if "MaliciousURLs" in scanner_name:
                            raise GuardrailViolation("Output contains a dangerous or untrusted link.", "MALICIOUS_URL")
                        elif "Sensitive" in scanner_name:
                            raise GuardrailViolation("Output contains restricted sensitive customer data.", "PII_DETECTED")
                        else:
                            raise GuardrailViolation("Output rejected by security scanner.", "SECURITY_VIOLATION")
                except GuardrailViolation:
                    raise
                except Exception as exc:
                    logger.error("Error executing LLM Guard output scanner %s: %s", scanner.__class__.__name__, exc)
                    raise GuardrailViolation("Security scanning error; output blocked for protection.", "SCANNER_ERROR") from exc
            return sanitized

        return response


_guard_manager = _LLMGuardManager()


def scan_input_security(query: str) -> str:
    """
    Public entry point for LLM Guard input security scanning.
    Verifies token limits, prompt injection, secrets, and PII.
    Returns cleaned / anonymized query or raises GuardrailViolation.
    """
    return _guard_manager.scan_input(query)


def scan_output_security(response: str, query: str = "") -> str:
    """
    Public entry point for LLM Guard output security scanning.
    Protects against secret leakage, PII exposure, and malicious URLs.
    Returns safe response or raises GuardrailViolation.
    """
    return _guard_manager.scan_output(response, prompt=query)


# 5. NeMo Guardrails Topic Control
class _NeMoTopicManager:
    """
    Manages NeMo Guardrails topic control for e-commerce scope enforcement.
    Ensures queries stay within allowed food, grocery, cart, order, and store bounds.
    """
    def __init__(self):
        self._rails = None
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            from nemoguardrails import RailsConfig, LLMRails
            config = RailsConfig.from_path("config/guardrails")
            self._rails = LLMRails(config)
            logger.info("NeMo Guardrails topic manager initialized.")
        except Exception as exc:
            logger.warning("NeMo Guardrails configuration could not be loaded: %s. Using heuristic topic classifier.", exc)
            self._rails = None

    def validate(self, query: str) -> bool:
        """
        Validates if query belongs to allowed e-commerce domains.
        Returns True if in-scope, or raises GuardrailViolation("OFF_TOPIC").
        """
        self._initialize()

        lowered = query.lower()

        # Deterministic Off-Topic Keywords
        off_topic_indicators = [
            "python script", "write code", "solve leetcode", "election",
            "presidential", "quantum physics", "legal advice", "sue my neighbor",
            "diagnose my fever", "prescribe antibiotic", "write a poem about love",
            "stock price of", "crypto trading"
        ]

        for off in off_topic_indicators:
            if off in lowered:
                logger.info("Topic Guardrail: OFF_TOPIC query intercepted: '%s'", off)
                raise GuardrailViolation(
                    "This assistant only handles food delivery, groceries, orders, and customer support. Please submit an e-commerce inquiry.",
                    "OFF_TOPIC"
                )

        # In-scope e-commerce indicators
        ecom_indicators = [
            "food", "order", "eat", "menu", "restaurant", "swiggy", "instamart",
            "cart", "price", "calorie", "protein", "delivery", "track", "refund",
            "cancel", "payment", "discount", "coupon", "grocery", "item", "dish",
            "healthy", "dinner", "lunch", "breakfast", "bread", "milk", "vegetable"
        ]

        if any(ecom in lowered for ecom in ecom_indicators):
            return True

        # If general greeting, allow
        if any(greet in lowered for greet in ["hello", "hi", "hey", "good morning", "good evening", "help"]):
            return True

        # NeMo Guardrails evaluation if active
        if self._rails:
            try:
                # Query NeMo Guardrails with context return to inspect explicit Colang state variables
                res = self._rails.generate(
                    messages=[{"role": "user", "content": query}],
                    options={"return_context": True}
                )
                context = res.get("context", {}) if isinstance(res, dict) else {}
                bot_reply = res.get("content", "") if isinstance(res, dict) else str(res)

                # Check explicit structured variable set in Colang ($off_topic = True)
                if (
                    context.get("off_topic") is True
                    or "cannot assist with that topic" in bot_reply.lower()
                    or "off topic" in bot_reply.lower()
                ):
                    logger.info("NeMo Guardrails structured rail blocked off-topic query.")
                    raise GuardrailViolation(
                        "This request is outside the supported e-commerce domain.",
                        "OFF_TOPIC"
                    )
            except GuardrailViolation:
                raise
            except Exception as exc:
                # Conscious degradation policy:
                # Security checks (secrets/PII) fail CLOSED (raise error).
                # Topic control fails OPEN to deterministic scope classification to avoid breaking valid shopper queries.
                logger.warning("NeMo topic engine unavailable (%s); falling back to deterministic scope rules.", exc)

        return True


_topic_manager = _NeMoTopicManager()


def validate_topic(query: str) -> bool:
    """
    Public entry point for NeMo topic control.
    Enforces that queries pertain strictly to food, groceries, orders, and store operations.
    """
    return _topic_manager.validate(query)


# 6. Deterministic Cart & Dietary Guardrails
def audit_cart_against_diet(
    cart: List[CartItem],
    dietary_spec: Optional[DietarySpec]
) -> Tuple[bool, Optional[str]]:
    """
    Strict Deterministic Egress Business Rule:
    Ensures that no items in the final cart violate allergy bounds,
    maximum calorie caps, or possess invalid/negative quantities.
    """
    if not cart:
        return True, None

    # Step 1: Validate physical integrity of all line items
    for item in cart:
        if item.quantity <= 0:
            return False, f"Invalid cart item quantity: '{item.item_name}' has quantity {item.quantity} (must be >= 1)."
        if item.price_per_unit <= 0.0:
            return False, f"Invalid price: '{item.item_name}' has non-positive price ₹{item.price_per_unit:.2f}."

    if not dietary_spec or dietary_spec.is_pass_through:
        return True, None

    # Step 2: Strict allergen detection (Title + Deep Ingredient Inspection)
    forbidden = [f.lower().strip() for f in dietary_spec.forbidden_ingredients if f.strip()]

    for item in cart:
        item_title = item.item_name.lower()
        item_ingredients = [ing.lower().strip() for ing in getattr(item, "ingredients", [])]

        for allergen in forbidden:
            # Word-boundary or substring allergen check in title
            if re.search(rf"\b{re.escape(allergen)}\b", item_title) or allergen in item_title:
                return False, f"Allergy violation: Item '{item.item_name}' contains restricted ingredient '{allergen}'."

            # Deep ingredient inspection (e.g. cookies containing peanut traces)
            for ing in item_ingredients:
                if re.search(rf"\b{re.escape(allergen)}\b", ing) or allergen in ing:
                    return False, f"Allergy violation: Item '{item.item_name}' contains restricted ingredient '{allergen}' in ingredients ({ing})."

    # Step 3: Strict calorie cap auditing (Zero Grace / 0% Tolerance)
    if dietary_spec.max_calories is not None:
        total_cals = 0

        for item in cart:
            if item.calories is None:
                # Strict requirement: unverified calorie data cannot be allowed in calorie-capped orders
                return False, f"Calorie compliance failure: Item '{item.item_name}' lacks verified nutritional calorie data for strict calorie-capped order."
            total_cals += item.calories * item.quantity

        # Strict hard cap: 0% grace threshold enforced
        if total_cals > dietary_spec.max_calories:
            return False, f"Calorie cap exceeded: Cart has {total_cals} kcal (strict limit: {dietary_spec.max_calories} kcal)."

    return True, None



# 7. Deterministic Financial Sanity Guardrails
def audit_financial_sanity(gross_total: float, discount: float) -> Tuple[bool, Optional[str]]:
    """
    High-Precision Deterministic Financial Guardrail:
    - Uses Python Decimal to prevent IEEE-754 floating point precision errors.
    - Rejects NaN, Infinity, negative values, and discounts exceeding gross total.
    """
    # 1. Float sanity / finite value verification
    if not (math.isfinite(gross_total) and math.isfinite(discount)):
        return False, "Invalid financial values: gross_total or discount contains NaN or Infinity."

    try:
        dec_gross = Decimal(str(gross_total))
        dec_discount = Decimal(str(discount))
    except (InvalidOperation, ValueError):
        return False, "Could not convert financial amounts to exact Decimal representation."

    # 2. Non-negative bounds
    if dec_gross < Decimal("0.00"):
        return False, f"Invalid gross total: ₹{dec_gross:.2f} cannot be negative."

    if dec_discount < Decimal("0.00"):
        return False, f"Invalid discount: ₹{dec_discount:.2f} cannot be negative."

    # 3. Discount cannot exceed gross total
    if dec_discount > dec_gross:
        return False, f"Discount (₹{dec_discount:.2f}) cannot exceed gross total (₹{dec_gross:.2f})."

    return True, None
