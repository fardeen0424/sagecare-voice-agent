"""Prints a link to talk to the agent from a browser: LiveKit Meet connected to this server's LiveKit.

Run: docker compose -f infra/docker-compose.yml exec agent python join.py
The token is valid for one hour and only for one new room; the agent joins it automatically.
"""

import os
import uuid
from datetime import timedelta
from urllib.parse import urlencode

from livekit import api

room = f"web-{uuid.uuid4().hex[:6]}"
token = (api.AccessToken().with_identity(f"browser-{uuid.uuid4().hex[:4]}").with_ttl(timedelta(hours=1))
         .with_grants(api.VideoGrants(room_join=True, room=room)).to_jwt())
url = f"wss://{os.environ['LIVEKIT_DOMAIN']}"
print(f"https://meet.livekit.io/custom?{urlencode({'liveKitUrl': url, 'token': token})}")
