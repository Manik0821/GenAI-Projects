import importlib.util
from pathlib import Path
import os
import sys

# Setup environment
PROJECT_ROOT = Path(r'C:\Users\mjain\Downloads\personal\GenAI-Projects')
BASE = PROJECT_ROOT / 'Travel-planner'

def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, BASE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    # Mocking gradio to avoid UI launch
    class Mock:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return Mock
    
    sys.modules['gradio'] = Mock()
    sys.modules['gradio.themes'] = Mock()
    
    # We need to manually load pf if app.py relies on it
    # But app.py loads it relative to its own path
    
    app_spec = importlib.util.spec_from_file_location('app', BASE / 'app.py')
    app = importlib.util.module_from_spec(app_spec)
    # Patch sys.path to allow imports in app.py if any
    sys.path.append(str(BASE))
    app_spec.loader.exec_module(app)

    def test_place(query):
        print(f'Testing: {query}')
        try:
            place = app.pf.geocode_place_details(query)
            if not place:
                print('No place details found.')
                return
            fa = place.get('formatted_address', 'No Address')
            print(f'Place: {fa}')
            photo_url = app._location_photo_url(place)
            if photo_url:
                print(f'Photo URL: {photo_url}')
            else:
                print('No photo URL found.')
        except Exception as e:
            import traceback
            traceback.print_exc()

    print('--- Starting Tests ---')
    test_place('Paris, France')
    test_place('Taj Mahal, Agra')
    print('--- Finished Tests ---')

except Exception as e:
    import traceback
    traceback.print_exc()
