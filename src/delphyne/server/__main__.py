"""
Server entry point.
"""

import uvicorn

from delphyne.server.basic_launcher import BasicLauncher
from delphyne.server.fastapi import make_server


# Look at http://0.0.0.0:8000/docs for API documentation
app = make_server(BasicLauncher())
uvicorn.run(app, host="0.0.0.0", port=8000)
