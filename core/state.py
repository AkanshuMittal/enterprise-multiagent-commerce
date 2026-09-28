"""
Enterprise Central State Register
Defines the single source of truth for the LangGraph multi-agent execution pipeline.
Uses TypedDict, Annotated reducers, Literal constraint flags, and Pydantic boundary models.
"""

import operator
from typing import Annotated, Dict, List, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ==============================================================================
# 1. Pydantic Boundary Models (Ingress, Tools & Webhook Schemas)
# ==============================================================================

class LocationSpec(BaseModel):
    """Spatial coordinates and human-readable locality context."""
    city: str = Field(default="Bangalore", description="Target delivery city.")
    locality: str = Field(default="Indiranagar", description="Locality or neighbourhood name.")
    lat: float = Field(default=12.9784, description="Latitude coordinate.")
    lng: float = Field(default=77.6408, description="Longitude coordinate.")


class UserProfile(BaseModel):
    """User profile data including health parameters and payment cards."""
    user_id: str = Field(..., description="Unique alphanumeric identifier of the user.")
    name: str = Field(default="Customer", description="Full name of the user.")
    location: LocationSpec = Field(default_factory=LocationSpec)
    allergies: List[str] = Field(default_factory=list, description="Declared allergen restrictions.")
    is_diabetic: bool = Field(default=False, description="Flag for sugar-restricted diet.")
    daily_calorie_target: Optional[int] = Field(default=2000, description="Daily target calories.")
    saved_cards: List[str] = Field(
        default_factory=lambda: ["HDFC Millennia", "ICICI Coral"],
        description="Saved payment bank card tiers for discount optimization."
    )


class DietarySpec(BaseModel):
    """Quantitative nutrition & allergy guardrail bounds produced by Node 2."""
    max_calories: Optional[int] = Field(None, description="Upper caloric limit for this meal.")
    min_protein_g: Optional[int] = Field(None, description="Minimum target protein in grams.")
    forbidden_ingredients: List[str] = Field(
        default_factory=list,
        description="Ingredients strictly disallowed due to allergies or health conditions."
    )
    lifestyle_tags: List[str] = Field(
        default_factory=list,
        description="Dietary lifestyle tags: e.g. 'high-protein', 'low-carb', 'weight-loss'."
    )
    is_pass_through: bool = Field(
        default=False,
        description="True if no special diet or medical restrictions apply (standard craving)."
    )


class CartItem(BaseModel):
    """Unified line item representing a food dish or grocery dark-store SKU."""
    item_id: str = Field(..., description="Unique product or dish SKU identifier.")
    item_name: str = Field(..., description="Display title of the item.")
    merchant_type: Literal["FOOD", "INSTAMART"] = Field(..., description="Merchant category.")
    source_name: str = Field(..., description="Restaurant name or Dark Store warehouse ID.")
    quantity: int = Field(default=1, ge=1, description="Quantity ordered.")
    price_per_unit: float = Field(..., gt=0.0, description="Price per unit in INR.")
    calories: Optional[int] = Field(None, description="Estimated calories per serving.")
    protein_g: Optional[int] = Field(None, description="Estimated protein grams.")


class SplitLedgerSummary(BaseModel):
    """Multi-merchant financial balance sheet compiled by Node 5."""
    food_subtotal: float = Field(default=0.0, ge=0.0)
    instamart_subtotal: float = Field(default=0.0, ge=0.0)
    gross_total: float = Field(default=0.0, ge=0.0)
    card_discount_applied: float = Field(default=0.0, ge=0.0)
    selected_card: Optional[str] = Field(None, description="Best discount card chosen.")
    delivery_fee: float = Field(default=0.0, ge=0.0)
    taxes: float = Field(default=0.0, ge=0.0)
    net_payable: float = Field(default=0.0, ge=0.0, description="Final amount charged.")


class TelemetryCoordinate(BaseModel):
    """Rider GPS telemetry coordinate streamed by Node 7."""
    lat: float
    lng: float
    rider_name: str
    vehicle_info: str
    eta_minutes: int
    status_text: str
    merchant_type: Literal["FOOD", "INSTAMART"]


# ==============================================================================
# 2. Custom Reducer Functions for State Merging
# ==============================================================================

def merge_cart_items(existing: List[CartItem], incoming: List[CartItem]) -> List[CartItem]:
    """
    Reducer function that merges items from multiple worker agents without duplicates.
    Updates quantities if the same item_id is re-added.
    """
    item_map = {item.item_id: item.model_copy() for item in (existing or [])}
    for item in (incoming or []):
        if item.item_id in item_map:
            item_map[item.item_id].quantity += item.quantity
        else:
            item_map[item.item_id] = item.model_copy()
    return list(item_map.values())


# ==============================================================================
# 3. Master Central State Register (TypedDict)
# ==============================================================================

class AgentState(TypedDict):
    """
    The master state ledger for the LangGraph StateGraph runtime.
    Maintains user context, agent outputs, financial ledgers, and checkpointer flags.
    """
    # Messaging & Thread Context
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    user_id: str
    active_stage: Literal[
        "SUPERVISOR_ROUTING",
        "DIET_ANALYSIS",
        "MERCHANT_SOURCING",
        "DISCOUNT_CALCULATION",
        "HITL_APPROVAL_GATE",
        "PAYMENT_AWAITING",
        "FULFILLMENT_ACTIVE",
        "ROLLBACK_RESOLVED",
        "COMPLETED",
        "ERROR"
    ]

    # Ingress Context
    user_profile: UserProfile
    user_raw_query: str

    # Agent Node Workspaces
    dietary_spec: Optional[DietarySpec]
    food_cart: Annotated[List[CartItem], merge_cart_items]
    grocery_cart: Annotated[List[CartItem], merge_cart_items]
    split_ledger: Optional[SplitLedgerSummary]

    # Human-in-the-Loop & Payment Safety Flags
    is_human_approved: bool
    is_payment_verified: bool
    payment_transaction_id: Optional[str]

    # Decoupled Telemetry & Rollback Audit Records
    telemetry_stream: Annotated[List[TelemetryCoordinate], operator.add]
    error_message: Optional[str]
    rollback_performed: bool
