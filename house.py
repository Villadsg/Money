import requests
import base64
import json
import time
from typing import Dict, List, Optional

class IdealistaAPI:
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
        self.base_url = "https://api.idealista.com"
        self.access_token = None
        self.token_expires_at = None
        
    def _encode_credentials(self) -> str:
        """Encode API key and secret for Basic authentication"""
        credentials = f"{self.api_key}:{self.secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return encoded
    
    def get_oauth_token(self) -> bool:
        """Get OAuth2 Bearer token using client credentials flow"""
        url = f"{self.base_url}/oauth/token"
        
        headers = {
            "Authorization": f"Basic {self._encode_credentials()}",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
        }
        
        data = {
            "grant_type": "client_credentials",
            "scope": "read"
        }
        
        try:
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            # Set expiry time (subtract 60 seconds for safety margin)
            self.token_expires_at = time.time() + token_data["expires_in"] - 60
            
            print(f"✓ OAuth token obtained successfully")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error obtaining OAuth token: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return False
    
    def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token"""
        if not self.access_token or (self.token_expires_at and time.time() >= self.token_expires_at):
            return self.get_oauth_token()
        return True
    
    def search_properties(self, search_params: Dict) -> Optional[Dict]:
        """Search for properties using the Idealista API"""
        if not self._ensure_valid_token():
            return None
            
        url = f"{self.base_url}/3.5/es/search"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, params=search_params)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error searching properties: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

def fetch_galicia_housing_data(api_key: str, secret: str) -> Dict:
    """
    Fetch comprehensive housing data for Galicia, Spain
    
    Galicia approximate center coordinates: 42.5751, -8.1339
    We'll use a large radius to cover the entire region
    """
    
    # Initialize API client
    api = IdealistaAPI(api_key, secret)
    
    # Galicia region coordinates (approximate center)
    galicia_center = "42.5751,-8.1339"
    # Distance to cover most of Galicia (approximately 150km radius)
    galicia_distance = 150000  # meters
    
    # Search parameters to get maximum data in one request
    search_params = {
        "center": galicia_center,
        "distance": galicia_distance,
        "operation": "sale",  # Start with sales data
        "propertyType": "homes",
        "maxItems": 50,  # Maximum allowed per request
        "numPage": 1,
        "locale": "es",
        "order": "publicationDate",
        "sort": "desc",
        "hasMultimedia": True,  # Get properties with photos/videos
        # Add filters to get comprehensive data
        "minSize": 60,  # Minimum size as per API
        "maxSize": 1000,  # Maximum size as per API
    }
    
    print("🏠 Fetching Galicia housing data...")
    print(f"📍 Search area: {galicia_center} (radius: {galicia_distance/1000}km)")
    print(f"🔍 Parameters: {search_params}")
    
    # Fetch sales data
    print("\n📊 Fetching sales data...")
    sales_data = api.search_properties(search_params)
    
    if sales_data:
        print(f"✓ Found {sales_data.get('total', 0)} sales properties")
        print(f"✓ Retrieved {len(sales_data.get('elementList', []))} properties in this page")
    
    # Also fetch rental data
    print("\n🏘️  Fetching rental data...")
    rental_params = search_params.copy()
    rental_params["operation"] = "rent"
    
    rental_data = api.search_properties(rental_params)
    
    if rental_data:
        print(f"✓ Found {rental_data.get('total', 0)} rental properties")
        print(f"✓ Retrieved {len(rental_data.get('elementList', []))} properties in this page")
    
    # Combine results
    results = {
        "search_area": {
            "region": "Galicia",
            "center": galicia_center,
            "radius_km": galicia_distance/1000,
            "country": "Spain"
        },
        "sales_data": sales_data,
        "rental_data": rental_data,
        "summary": {
            "total_sales_properties": sales_data.get('total', 0) if sales_data else 0,
            "total_rental_properties": rental_data.get('total', 0) if rental_data else 0,
            "sales_retrieved": len(sales_data.get('elementList', [])) if sales_data else 0,
            "rentals_retrieved": len(rental_data.get('elementList', [])) if rental_data else 0,
        }
    }
    
    return results

def analyze_property_data(results: Dict) -> None:
    """Analyze and display key insights from the property data"""
    print("\n" + "="*60)
    print("📈 GALICIA HOUSING MARKET ANALYSIS")
    print("="*60)
    
    summary = results.get("summary", {})
    
    print(f"🏠 Total Properties Found:")
    print(f"   • Sales: {summary.get('total_sales_properties', 0):,}")
    print(f"   • Rentals: {summary.get('total_rental_properties', 0):,}")
    
    print(f"\n📄 Retrieved in this request:")
    print(f"   • Sales: {summary.get('sales_retrieved', 0)}")
    print(f"   • Rentals: {summary.get('rentals_retrieved', 0)}")
    
    # Analyze sales data
    if results.get("sales_data") and results["sales_data"].get("elementList"):
        sales_props = results["sales_data"]["elementList"]
        prices = [p.get("price", 0) for p in sales_props if p.get("price")]
        sizes = [p.get("size", 0) for p in sales_props if p.get("size")]
        
        if prices:
            print(f"\n💰 Sales Price Analysis:")
            print(f"   • Average: €{sum(prices)/len(prices):,.0f}")
            print(f"   • Min: €{min(prices):,}")
            print(f"   • Max: €{max(prices):,}")
        
        if sizes:
            print(f"\n📏 Property Size Analysis:")
            print(f"   • Average: {sum(sizes)/len(sizes):.0f} m²")
            print(f"   • Min: {min(sizes)} m²")
            print(f"   • Max: {max(sizes)} m²")
    
    # Analyze rental data
    if results.get("rental_data") and results["rental_data"].get("elementList"):
        rental_props = results["rental_data"]["elementList"]
        rents = [p.get("price", 0) for p in rental_props if p.get("price")]
        
        if rents:
            print(f"\n🏡 Rental Price Analysis:")
            print(f"   • Average: €{sum(rents)/len(rents):,.0f}/month")
            print(f"   • Min: €{min(rents):,}/month")
            print(f"   • Max: €{max(rents):,}/month")

def save_results(results: Dict, filename: str = "galicia_housing_data.json") -> None:
    """Save results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {filename}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")

if __name__ == "__main__":
    # Your API credentials
    API_KEY = "p9rstzhmdl0tl23qthoy4dqld02cdz5y"
    SECRET = "fpDudSAto8dz"
    
    print("🚀 Starting Galicia Housing Data Fetch...")
    print(f"🔑 Using API Key: {API_KEY}")
    
    # Fetch the data
    results = fetch_galicia_housing_data(API_KEY, SECRET)
    
    if results:
        # Analyze and display results
        analyze_property_data(results)
        
        # Save results to file
        save_results(results)
        
        print(f"\n✅ Process completed successfully!")
        print(f"📊 Use the saved JSON file for further analysis")
    else:
        print("❌ Failed to fetch data")