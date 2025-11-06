"""
Booking.com flight search data source implementation
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

from .base import BaseAPI

logger = logging.getLogger("booking_source")


class BookingSource(BaseAPI):
    """Booking.com data source"""

    def __init__(self, config: Dict[str, Any], proxy_url: Optional[str] = None):
        """Initialize Booking.com API data source"""

        self._timeout = config["timeout"]
        self.proxy_url = config["external_api_proxy_url"]
        if proxy_url:
            self.proxy_url = proxy_url
        self.headers = {
            "X-Original-Host": config["booking_base_url"],
            "X-Biz-Id": "matrix-agent",
            "X-Request-Timeout": str(config["timeout"] - 5),
        }

    @property
    def source_name(self) -> str:
        """
        Get data source name

        Returns:
            str: Data source name
        """
        return "booking"

    def get_api_info(self) -> Dict[str, Any]:
        """
        Get basic information about the data source

        Returns:
            Dict[str, Any]: Contains basic information like data source name and version
        """
        return {
            "name": self.source_name,
            "description": "Booking.com data source, providing flight search and hotel search services",
        }

    async def search_flights(
        self,
        from_code: str,
        to_code: str,
        depart_date: str,
        return_date: Optional[str] = None,
        stops: str = "none",
        page_no: int = 1,
        adults: int = 1,
        children: Optional[str] = None,
        sort: str = "BEST",
        cabin_class: str = "ECONOMY",
        currency_code: str = "USD",
    ) -> Dict[str, Any]:
        """
        Search for flights

        Args:
            from_code(str): Departure airport code, e.g.: PEK
            to_code(str): Destination airport code, e.g.: CAN
            depart_date(str): Departure date, format: YYYY-MM-DD
            return_date(Optional[str]): Return date, format: YYYY-MM-DD (optional)
            stops(str): Number of stops, options: none, 0, 1, 2
            page_no(int): Page number, default is 1
            adults(int): Number of adults, default is 1
            children(Optional[str]): Children's ages, comma separated, e.g.: 0,17 (optional)
            sort(str): Sort method, options: BEST, CHEAPEST, FASTEST
            cabin_class(str): Cabin class, options: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
            currency_code(str): Currency code, default USD

        Returns:
            Dict[str, Any]: Dictionary containing flight search results, e.g.
            {
                "success": True,                   # Whether successful
                "data": {                          # If successful, contains the following fields
                    "flights": [                   # Flight list
                        {
                            "stops": 0,            # Number of stops
                            "segments": [          # Segment information
                                {
                                    "flight_number": "CA1385",  # Flight number
                                    "from": "PEK", # Departure airport
                                    "to": "CAN",   # Arrival airport
                                    "departure": "2025-04-19T20:05:00",  # Departure time
                                    "arrival": "2025-04-19T23:10:00",     # Arrival time
                                    "total_time": 3.08  # Segment flight time
                                },
                                {
                                    "flight_number": "CA1386",
                                    "from": "CAN",
                                    "to": "PEK",
                                    "departure": "2025-04-26T06:25:00",
                                    "arrival": "2025-04-26T09:20:00",
                                    "total_time": 2.92  # Segment flight time
                                }
                            ],
                            "price": {             # Price information
                                "currency": "CNY", # Currency
                                "amount": 14272.26 # Total price
                            },
                            "total_time": 6.00  # Total flight time
                        }
                    ]
                }
            }
        """
        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Starting flight search")
        #     >>> result = await client.booking.search_flights(
        #     ...     from_code="PEK",
        #     ...     to_code="CAN",
        #     ...     depart_date="2025-04-19",
        #     ...     return_date="2025-04-26",
        #     ...     cabin_class="ECONOMY"
        #     ... )
        #     >>> if not result["success"]:
        #     ...     print(f"Search failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Search successful")
        # """
        try:
            # Build request parameters
            params = {
                "fromId": f"{from_code}.AIRPORT",
                "toId": f"{to_code}.AIRPORT",
                "departDate": depart_date,
                "stops": stops,
                "pageNo": page_no,
                "adults": adults,
                "sort": sort,
                "cabinClass": cabin_class,
                "currency_code": currency_code,
            }

            # Add optional parameters
            if return_date:
                params["returnDate"] = return_date
            if children:
                params["children"] = children

            request_url = f"{self.proxy_url}/api/v1/flights/searchFlights"

            logger.info("Starting flight search")

            # Send request
            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                        # Check response status
                        response.raise_for_status()
                        data = await response.json()

            except asyncio.TimeoutError:
                error_msg = f"Request timeout (timeout={self._timeout}s)"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            except aiohttp.ClientError as e:
                error_msg = f"Request failed: {str(e)}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            # Check if API response has error
            if not data.get("status"):
                error_msg = data.get("message", "Unknown error")
                logger.error(f"API returned error: {error_msg}")
                return {"success": False, "error": error_msg}

            # 检查是否存在错误
            if not data.get("data", {}).get("flightOffers"):
                logger.error("No flight offers found")
                return {"success": True, "data": {"flights": []}}

            # Simplify response data structure
            simplified_flights = []
            for offer in data["data"]["flightOffers"]:
                legs_info = []
                stops_count = 0

                total_time = 0
                for segment in offer["segments"]:
                    # Get flight number and stop info
                    for leg in segment["legs"]:
                        flight_number = f"{leg['flightInfo']['carrierInfo']['marketingCarrier']}{leg['flightInfo']['flightNumber']}"
                        # Count stops
                        stops_count += len(leg.get("flightStops", []))

                        # Add segment info
                        legs_info.append(
                            {
                                "flight_number": flight_number,
                                "from": leg["departureAirport"]["code"],
                                "to": leg["arrivalAirport"]["code"],
                                "departure": leg["departureTime"],
                                "arrival": leg["arrivalTime"],
                                "total_time": self._format_duration(leg["totalTime"]),  # Segment flight time
                            }
                        )
                        total_time += leg["totalTime"]
                # Handle price
                price = offer["priceBreakdown"]["total"]
                total_amount = float(price["units"]) + float(price["nanos"]) / 1_000_000_000

                simplified_flights.append(
                    {
                        "stops": stops_count,
                        "segments": legs_info,
                        "total_time": self._format_duration(total_time),
                        "price": {"currency": price["currencyCode"], "amount": total_amount},
                    }
                )

            return {"success": True, "data": {"flights": simplified_flights}}

        except Exception as e:
            error_msg = f"Error occurred while searching flights: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    async def _search_hotel_destinations(self, query: str) -> Dict[str, Any]:
        """
        Search for hotel destinations

        Args:
            query(str): Search keyword, e.g.: shanghai

        Returns:
            Dict[str, Any]: Dictionary containing destination search results, e.g.
            {
                "success": True,                   # Whether successful
                "data": {                          # If successful, contains the following fields
                    "destinations": [              # Destination list
                        {
                            "dest_id": "-1924465", # Destination ID
                            "search_type": "city",   # Destination type
                            "name": "Shanghai",    # Destination name
                            "city_name": "Shanghai", # City name
                            "label": "Shanghai, Shanghai Area, China", # Full label
                            "longitude": 121.4763, # Longitude
                            "latitude": 31.229422, # Latitude
                            "country": "China"     # Country
                        }
                    ]
                }
            }
        """
        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Starting destination search")
        #     >>> result = await client.booking.search_destinations("shanghai")
        #     >>> if not result["success"]:
        #     ...     print(f"Search failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Search successful")
        # """
        try:
            # 构建请求参数
            params = {"query": query}

            request_url = f"{self.proxy_url}/api/v1/hotels/searchDestination"

            logger.info("Starting destination search")

            # 发送请求
            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                        # 检查响应状态
                        response.raise_for_status()
                        data = await response.json()

            except asyncio.TimeoutError:
                error_msg = f"Request timeout (timeout={self._timeout}s)"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            except aiohttp.ClientError as e:
                error_msg = f"Request failed: {str(e)}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            # 检查API响应中是否有错误
            if not data.get("status"):
                error_msg = data.get("message", "Unknown error")
                logger.error(f"API returned error: {error_msg}")
                return {"success": False, "error": error_msg}

            # 简化响应数据结构
            simplified_destinations = []
            for dest in data["data"]:
                simplified_destinations.append(
                    {
                        "dest_id": dest["dest_id"],
                        "search_type": dest["search_type"],
                        "name": dest["name"],
                        "city_name": dest["city_name"],
                        "label": dest["label"],
                        "longitude": dest["longitude"],
                        "latitude": dest["latitude"],
                        "country": dest["country"],
                    }
                )

            return {"success": True, "data": {"destinations": simplified_destinations}}

        except Exception as e:
            error_msg = f"Error occurred while searching destinations: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    def _format_duration(self, seconds: int) -> str:
        """Convert seconds to hours and minutes format"""
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours} hours {mins} minutes"


if __name__ == "__main__":
    import json

    from external_api.data_sources.client import get_client

    async def main():
        client = get_client()
        result = await client.booking.search_flights(
            from_code="PEK", to_code="CAN", depart_date="2025-04-19", return_date="2025-04-26", cabin_class="ECONOMY"
        )
        print(json.dumps(result, indent=4))

    asyncio.run(main())