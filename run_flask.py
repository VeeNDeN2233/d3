
import os
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    
    print(f"Запуск Flask приложения на http://{host}:{port}")
    print(f"Debug режим: {debug}")
    
    app.run(host=host, port=port, debug=debug)
