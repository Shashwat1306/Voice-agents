import logging
import json
from pathlib import Path
from typing import Annotated, Optional
from datetime import datetime
import uuid
from pydantic import Field

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Load product catalog

def load_catalog():
    """Load the product catalog from JSON file"""
    content_path = Path("shared-data/product_catalog.json")
    with open(content_path, "r") as f:
        data = json.load(f)
        return data['products']

# Load/save orders
def load_orders():
    """Load existing orders from JSON file"""
    orders_path = Path("shared-data/orders.json")
    if orders_path.exists():
        with open(orders_path, "r") as f:
            return json.load(f)
    return []

def save_orders(orders):
    """Save orders to JSON file"""
    orders_path = Path("shared-data/orders.json")
    with open(orders_path, "w") as f:
        json.dump(orders, f, indent=2)


class EcommerceAgent(Agent):
    """Voice Shopping Assistant following Agentic Commerce Protocol patterns"""
    
    def __init__(self) -> None:
        # Load product catalog and orders
        self.catalog = load_catalog()
        self.all_orders = load_orders()
        
        # Session-specific data
        self.shopping_cart = []
        self.last_shown_products = []  # Track products mentioned to user
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        super().__init__(
            instructions="""You are a helpful voice shopping assistant for an e-commerce store.

🛍️ YOUR ROLE:
You help customers discover products, answer questions about items, and complete purchases through natural conversation.

📋 PRODUCT CATEGORIES WE CARRY:
- Coffee Mugs & Drinkware (mugs, bottles)
- Clothing (t-shirts, hoodies)
- Bags & Backpacks
- Stationery (notebooks, journals)

💬 CONVERSATION STYLE:
- Friendly, helpful, and enthusiastic about our products
- Use natural language, not robotic catalog descriptions
- Confirm important details (size, color, quantity) before ordering
- Keep responses concise (2-3 sentences max)

🔍 BROWSING PRODUCTS:
When customers ask to see products:
1. Use search_catalog to find matching items
2. Present 2-3 relevant products at a time (don't overwhelm!)
3. Include: name, price, and 1 key feature
4. Remember what you showed them for follow-up questions

Examples:
- "Show me coffee mugs" → Call search_catalog(category="mug")
- "Do you have black hoodies under 2000?" → Call search_catalog(category="clothing", product_type="hoodie", color="black", max_price=2000)
- "What colors does that mug come in?" → Reference last_shown_products, call search_catalog for variants

🛒 ADDING TO CART:
When customer wants to buy:
1. Identify which product (use product_id from last_shown_products)
2. Confirm size/color if it's clothing
3. Call add_to_cart with product_id and quantity
4. Confirm addition: "Added [product name] to your cart!"

Handle references intelligently:
- "I'll take the second one" → Use last_shown_products[1]
- "Add the black hoodie" → Find matching product from recent results
- "Give me two of those mugs" → quantity=2

📦 PLACING ORDERS:
When customer is ready to checkout:
1. Call view_cart to show current cart
2. Confirm with customer
3. Call create_order to finalize
4. Read back order ID and total

🎯 KEY RULES:
1. ALWAYS search catalog before discussing products (don't make up items!)
2. Track products you mention in conversation for easy reference
3. Confirm details before adding to cart (especially size for clothing)
4. Keep cart visible - remind customer what's in cart when relevant
5. Be proactive: suggest related items, mention deals
6. Handle ambiguity: ask clarifying questions if customer request is unclear

🚫 DON'T:
- List more than 3-4 products at once
- Make up products not in catalog
- Add items to cart without confirmation
- Skip size confirmation for clothing
- Be pushy or sales-y

REMEMBER: This is voice conversation - be natural, concise, and helpful!
""",
        )
    
    @function_tool
    async def search_catalog(
        self,
        category: Annotated[Optional[str], Field(default=None, description="Product category: 'mug', 'clothing', 'bottle', 'bag', 'stationery'")],
        product_type: Annotated[Optional[str], Field(default=None, description="Specific type: 't-shirt', 'hoodie', 'notebook'")],
        color: Annotated[Optional[str], Field(default=None, description="Color filter: 'white', 'black', 'blue', 'grey', etc.")],
        max_price: Annotated[Optional[int], Field(default=None, description="Maximum price in INR")],
        min_price: Annotated[Optional[int], Field(default=None, description="Minimum price in INR")],
        keyword: Annotated[Optional[str], Field(default=None, description="Search keyword in name or description")],
    ):
        """
        Search the product catalog with filters. Returns list of matching products.
        Use this whenever customer asks about products or wants to browse.
        """
        logger.info(f"Searching catalog: category={category}, type={product_type}, color={color}, price={min_price}-{max_price}, keyword={keyword}")
        
        results = []
        
        for product in self.catalog:
            # Apply filters
            if category and product.get('category') != category:
                continue
            
            if product_type and product.get('type') != product_type:
                continue
            
            if color:
                product_color = product.get('color', '').lower()
                if color.lower() not in product_color:
                    continue
            
            if max_price and product.get('price', 0) > max_price:
                continue
            
            if min_price and product.get('price', 0) < min_price:
                continue
            
            if keyword:
                keyword_lower = keyword.lower()
                searchable = f"{product.get('name', '')} {product.get('description', '')}".lower()
                if keyword_lower not in searchable:
                    continue
            
            results.append(product)
        
        # Store results for later reference
        self.last_shown_products = results[:10]  # Keep top 10 for reference
        
        if not results:
            return {
                "success": True,
                "count": 0,
                "message": "No products found matching those criteria",
                "products": []
            }
        
        return {
            "success": True,
            "count": len(results),
            "products": results,
            "message": f"Found {len(results)} matching product(s)"
        }
    
    @function_tool
    async def get_product_details(
        self,
        product_id: Annotated[str, "Product ID to get detailed information"]
    ):
        """
        Get complete details for a specific product by ID.
        Use when customer asks detailed questions about a specific item.
        """
        logger.info(f"Getting product details for: {product_id}")
        
        for product in self.catalog:
            if product['id'] == product_id:
                return {
                    "success": True,
                    "product": product
                }
        
        return {
            "success": False,
            "message": f"Product {product_id} not found"
        }
    
    @function_tool
    async def add_to_cart(
        self,
        product_id: Annotated[str, "Product ID to add to cart"],
        quantity: Annotated[int, "Quantity to add"] = 1,
        size: Annotated[str, "Size for clothing items (S, M, L, XL, XXL)"] = None,
    ):
        """
        Add a product to the shopping cart.
        For clothing, ALWAYS ask for size before calling this.
        """
        logger.info(f"Adding to cart: {product_id}, quantity={quantity}, size={size}")
        
        # Find the product
        product = None
        for p in self.catalog:
            if p['id'] == product_id:
                product = p
                break
        
        if not product:
            return {
                "success": False,
                "message": f"Product {product_id} not found in catalog"
            }
        
        # Check if size is required
        if product.get('category') == 'clothing' and not size:
            available_sizes = product.get('sizes', [])
            return {
                "success": False,
                "message": f"Size required for clothing. Available sizes: {', '.join(available_sizes)}",
                "requires_size": True,
                "available_sizes": available_sizes
            }
        
        # Validate size if provided
        if size and product.get('sizes'):
            if size.upper() not in product.get('sizes', []):
                return {
                    "success": False,
                    "message": f"Size {size} not available. Available sizes: {', '.join(product.get('sizes', []))}",
                    "available_sizes": product.get('sizes', [])
                }
        
        # Add to cart
        cart_item = {
            "product_id": product_id,
            "name": product['name'],
            "price": product['price'],
            "currency": product['currency'],
            "quantity": quantity,
            "size": size.upper() if size else None
        }
        
        self.shopping_cart.append(cart_item)
        
        subtotal = product['price'] * quantity
        
        return {
            "success": True,
            "message": f"Added {quantity}x {product['name']} to cart",
            "cart_item": cart_item,
            "item_subtotal": subtotal,
            "cart_total": sum(item['price'] * item['quantity'] for item in self.shopping_cart)
        }
    
    @function_tool
    async def view_cart(self):
        """
        View current shopping cart contents and total.
        Use when customer asks "what's in my cart?" or before checkout.
        """
        logger.info("Viewing shopping cart")
        
        if not self.shopping_cart:
            return {
                "success": True,
                "empty": True,
                "message": "Your cart is empty",
                "items": [],
                "total": 0
            }
        
        total = sum(item['price'] * item['quantity'] for item in self.shopping_cart)
        
        return {
            "success": True,
            "empty": False,
            "items": self.shopping_cart,
            "item_count": len(self.shopping_cart),
            "total": total,
            "currency": "INR"
        }
    
    @function_tool
    async def remove_from_cart(
        self,
        index: Annotated[int, "Index of item to remove (0-based, from view_cart results)"]
    ):
        """
        Remove an item from shopping cart by index.
        Call view_cart first to show customer the cart with indices.
        """
        logger.info(f"Removing item at index {index} from cart")
        
        if index < 0 or index >= len(self.shopping_cart):
            return {
                "success": False,
                "message": f"Invalid index. Cart has {len(self.shopping_cart)} items (indices 0-{len(self.shopping_cart)-1})"
            }
        
        removed_item = self.shopping_cart.pop(index)
        
        return {
            "success": True,
            "message": f"Removed {removed_item['name']} from cart",
            "removed_item": removed_item,
            "cart_total": sum(item['price'] * item['quantity'] for item in self.shopping_cart)
        }
    
    @function_tool
    async def create_order(self):
        """
        Create an order from current shopping cart.
        This finalizes the purchase and clears the cart.
        Call view_cart first to confirm with customer before creating order.
        """
        logger.info("Creating order from shopping cart")
        
        if not self.shopping_cart:
            return {
                "success": False,
                "message": "Cannot create order - cart is empty"
            }
        
        # Generate order
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
        
        total = sum(item['price'] * item['quantity'] for item in self.shopping_cart)
        
        order = {
            "order_id": order_id,
            "session_id": self.session_id,
            "items": self.shopping_cart.copy(),
            "total": total,
            "currency": "INR",
            "created_at": datetime.now().isoformat(),
            "status": "confirmed"
        }
        
        # Save order
        self.all_orders.append(order)
        save_orders(self.all_orders)
        
        # Clear cart
        cart_copy = self.shopping_cart.copy()
        self.shopping_cart = []
        
        return {
            "success": True,
            "order": order,
            "message": f"Order {order_id} created successfully!",
            "order_id": order_id,
            "total": total,
            "items_purchased": len(cart_copy)
        }
    
    @function_tool
    async def get_last_order(self):
        """
        Get details of the most recent order from this session.
        Use when customer asks "what did I just buy?" or "show my last order".
        """
        logger.info("Fetching last order")
        
        # Find orders from this session
        session_orders = [o for o in self.all_orders if o.get('session_id') == self.session_id]
        
        if not session_orders:
            return {
                "success": False,
                "message": "No orders found in this session"
            }
        
        # Get most recent
        last_order = session_orders[-1]
        
        return {
            "success": True,
            "order": last_order
        }


async def prewarm(proc: JobProcess):
    """Prewarm the model and resources before handling requests"""
    await proc.userdata  # Wait for user data to be ready


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    logger.info("Starting E-commerce Shopping Assistant agent")

    # Create session
    session = AgentSession(
        llm=google.llm.LLM(model="gemini-2.5-flash"),
        stt=deepgram.STT(model="nova-3"),
        tts=murf.TTS(voice="en-US-matthew", model="FALCON"),
    )

    # Track usage metrics
    usage_collector = metrics.UsageCollector()

    @session.on(MetricsCollectedEvent)
    def _on_metrics_collected(event: MetricsCollectedEvent):
        metrics.log_metrics(event.metrics)
        usage_collector.collect(event.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=EcommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))