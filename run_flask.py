"""
Скрипт запуска Flask приложения.
"""

import os
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"Запуск Flask приложения на http://127.0.0.1:{port}")
    print(f"Debug режим: {debug}")
    
    app.run(host='127.0.0.1', port=port, debug=debug)
