import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from .api import app, miro_ids

client = TestClient(app)


def test_health_check():
    response = client.get('/palette_api/health_check')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}


def test_docs():
    response = client.get('/palette_api/docs')
    assert response.status_code == 200


def test_redoc():
    response = client.get('/palette_api/redoc')
    assert response.status_code == 200
