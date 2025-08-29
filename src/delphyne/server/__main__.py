"""
Server entry point.

Usage: python -m delphyne.server
"""

import uvicorn

from delphyne.server.basic_launcher import BasicLauncher
from delphyne.server.fastapi import make_server


def main(port: int = 3008):
    # Look at http://0.0.0.0:3008/docs for API documentation
    app = make_server(BasicLauncher())
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
