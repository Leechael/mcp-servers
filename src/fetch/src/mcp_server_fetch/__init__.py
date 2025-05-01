from .server import serve
import argparse
import asyncio
import os


def main():
    """MCP Fetch Server - HTTP fetching functionality for MCP"""
    parser = argparse.ArgumentParser(description="MCP Fetch Server")
    parser.add_argument("--user-agent", help="Custom User-Agent string")
    parser.add_argument("--ignore-robots-txt", action="store_true", help="Ignore robots.txt restrictions")
    parser.add_argument("--proxy-url", help="Proxy URL to use for requests")
    args = parser.parse_args()

    # Get configuration from environment variables if not provided via CLI
    user_agent = args.user_agent or os.environ.get("MCP_FETCH_USER_AGENT")
    ignore_robots_txt = args.ignore_robots_txt or os.environ.get("MCP_FETCH_IGNORE_ROBOTS_TXT", "").lower() in ("1", "true", "yes")
    proxy_url = args.proxy_url or os.environ.get("MCP_FETCH_PROXY_URL")

    # Create a new event loop and run the server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(serve(user_agent, ignore_robots_txt, proxy_url))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
