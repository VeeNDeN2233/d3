"""
Flask приложение для системы анализа движений младенцев.
Реализует полноценную многостраничность для избежания проблем с видимостью компонентов.
"""

# Устанавливаем backend для matplotlib перед любыми импортами, которые могут использовать matplotlib
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend для серверного окружения

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, flash
from werkzeug.utils import secure_filename
import threading
import zipfile
import io
from datetime import datetime

# Импорт существующих модулей
from auth.auth_manager import AuthManager
from core.state_manager import StateManager, AnalysisStep
from core.auth_handler import AuthHandler
from utils.ui_state_manager import UIStateManager
from utils.logger_config import setup_logging, get_log_entries, admin_logger, LOG_FILE, ERROR_LOG_FILE, ADMIN_LOG_FILE

# Импорт логики анализа
from flask_analysis import (
    load_models_for_flask,
    analyze_video_flask,
    get_models_status,
    _model,
    _detector,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем Flask приложение
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Создаем директории
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path('templates').mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)

# Менеджеры
auth_manager = AuthManager()
_state_manager = StateManager()
_auth_handler = AuthHandler()
ui_state_manager = UIStateManager(_state_manager)


def require_auth():
    """Декоратор для проверки аутентификации."""
    def decorator(f):
        def wrapper(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator


def require_admin():
    """Декоратор для проверки прав администратора."""
    def decorator(f):
        def wrapper(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            
            user_role = session.get('role', 'user')
            if user_role != 'admin':
                flash('Доступ запрещен. Требуются права администратора.', 'error')
                return redirect(url_for('main'))
            
            return f(*args, **kwargs)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator


@app.route('/')
def index():
    """Главная страница - редирект на login или main."""
    if 'user_id' in session:
        return redirect(url_for('main'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Страница входа."""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        if not email or not password:
            flash('Заполните все поля', 'error')
            return render_template('login.html')
        
        # Используем AuthHandler для входа
        success, message, user_data, session_token = _auth_handler.login(email, password)
        
        if success and user_data and session_token:
            # Сохраняем в Flask session
            session['user_id'] = user_data.get('id')
            session['email'] = user_data.get('email')
            session['username'] = user_data.get('username')
            session['full_name'] = user_data.get('full_name')
            session['role'] = user_data.get('role', 'user')
            session['session_token'] = session_token
            
            # Логируем вход
            if user_data.get('role') == 'admin':
                admin_logger.info(f"Администратор {user_data.get('email')} вошел в систему")
            else:
                logger.info(f"Пользователь {user_data.get('email')} вошел в систему")
            
            # Обновляем StateManager
            _state_manager.update_user(
                is_authenticated=True,
                session_token=session_token,
                email=user_data.get('email'),
                username=user_data.get('username'),
                full_name=user_data.get('full_name'),
                role=user_data.get('role', 'user'),
            )
            
            flash('Вход выполнен успешно', 'success')
            return redirect(url_for('main'))
        else:
            flash(message or 'Неверное имя пользователя или пароль', 'error')
    
    # Если уже авторизован, редирект на главную
    if 'user_id' in session:
        return redirect(url_for('main'))
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Страница регистрации."""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        password_confirm = request.form.get('password_confirm', '').strip()
        full_name = request.form.get('full_name', '').strip()
        
        if not email or not password:
            flash('Заполните email и пароль', 'error')
            return render_template('register.html')
        
        if password != password_confirm:
            flash('Пароли не совпадают', 'error')
            return render_template('register.html')
        
        # Используем AuthHandler для регистрации
        success, message, user_data, session_token = _auth_handler.register(
            email, password, password_confirm, full_name if full_name else None
        )
        
        if success and session_token and user_data:
            # Сохраняем в Flask session
            session['user_id'] = user_data.get('id')
            session['email'] = user_data.get('email')
            session['username'] = user_data.get('username')
            session['full_name'] = user_data.get('full_name')
            session['role'] = user_data.get('role', 'user')
            session['session_token'] = session_token
            
            # Логируем регистрацию
            logger.info(f"Новый пользователь зарегистрирован: {user_data.get('email')}")
            
            # Обновляем StateManager
            _state_manager.update_user(
                is_authenticated=True,
                session_token=session_token,
                email=user_data.get('email'),
                username=user_data.get('username'),
                full_name=user_data.get('full_name'),
                role=user_data.get('role', 'user'),
            )
            
            flash('Регистрация выполнена успешно', 'success')
            return redirect(url_for('main'))
        else:
            flash(message or 'Ошибка регистрации', 'error')
    
    # Если уже авторизован, редирект на главную
    if 'user_id' in session:
        return redirect(url_for('main'))
    
    return render_template('register.html')


@app.route('/logout')
def logout():
    """Выход из системы."""
    session_token = session.get('session_token')
    if session_token:
        auth_manager.logout(session_token)
    
    # Очищаем Flask session
    session.clear()
    
    # Обновляем StateManager
    _state_manager.update_user(is_authenticated=False, session_token=None)
    
    flash('Выход выполнен', 'info')
    return redirect(url_for('login'))


@app.route('/main')
@require_auth()
def main():
    """Главная страница с анализом."""
    # Проверяем, загружены ли модели
    models_loaded = _model is not None and _detector is not None
    
    return render_template(
        'main.html',
        user_email=session.get('email'),
        user_name=session.get('full_name') or session.get('username'),
        models_loaded=models_loaded,
    )


@app.route('/api/load_models', methods=['POST'])
@require_auth()
def api_load_models():
    """API endpoint для загрузки моделей."""
    try:
        success, message = load_models_for_flask()
        
        return jsonify({
            'success': success,
            'message': message,
            'models_loaded': success,
        })
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Ошибка загрузки моделей: {str(e)}',
            'models_loaded': False,
        }), 500


@app.route('/api/upload_video', methods=['POST'])
@require_auth()
def api_upload_video():
    """API endpoint для загрузки видео."""
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'Видео не загружено'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Файл не выбран'}), 400
    
    try:
        # Сохраняем файл
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)
        
        # Сохраняем путь в session
        session['uploaded_video'] = str(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Видео загружено успешно',
            'filename': filename,
        })
    except Exception as e:
        logger.error(f"Ошибка загрузки видео: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Ошибка загрузки видео: {str(e)}',
        }), 500


@app.route('/api/analyze', methods=['POST'])
@require_auth()
def api_analyze():
    """API endpoint для анализа видео."""
    if 'uploaded_video' not in session:
        return jsonify({'success': False, 'message': 'Видео не загружено'}), 400
    
    video_path = Path(session['uploaded_video'])
    if not video_path.exists():
        return jsonify({'success': False, 'message': 'Файл видео не найден'}), 404
    
    age_weeks = request.json.get('age_weeks', 12)
    gestational_age = request.json.get('gestational_age', 40)
    
    models_status = get_models_status()
    if not models_status['loaded']:
        return jsonify({
            'success': False,
            'message': 'Модели не загружены. Пожалуйста, загрузите модели сначала.',
        }), 400
    
    try:
        # Выполняем анализ
        success, plot_path, video_path_result, report_text, error_msg, output_dir_path = analyze_video_flask(
            video_path=video_path,
            age_weeks=age_weeks,
            gestational_age_weeks=gestational_age,
        )
        
        if not success:
            return jsonify({
                'success': False,
                'message': error_msg or 'Ошибка анализа',
            }), 500
        
        # Преобразуем пути в URL для Flask
        # plot_path и video_path_result уже относительные от results/
        plot_url = f"/results/{plot_path}" if plot_path else None
        video_url = f"/results/{video_path_result}" if video_path_result else None
        
        # Сохраняем путь к директории результатов в сессии для скачивания
        if output_dir_path:
            session['last_analysis_dir'] = output_dir_path
        
        return jsonify({
            'success': True,
            'plot_path': plot_url,
            'video_path': video_url,
            'report_text': report_text,
            'output_dir': output_dir_path,
        })
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Ошибка анализа: {str(e)}',
        }), 500


@app.route('/api/check_status')
@require_auth()
def api_check_status():
    """API endpoint для проверки статуса системы."""
    models_status = get_models_status()
    return jsonify({
        'authenticated': True,
        'models_loaded': models_status['loaded'],
        'user_email': session.get('email'),
    })


@app.route('/results/<path:filename>')
@require_auth()
def serve_result(filename):
    """Сервирование результатов анализа."""
    results_dir = Path('results')
    # Безопасно обрабатываем путь, предотвращая path traversal
    file_path = (results_dir / filename).resolve()
    results_dir_resolved = results_dir.resolve()
    
    # Проверяем, что файл находится внутри results/
    if not str(file_path).startswith(str(results_dir_resolved)):
        return "Access denied", 403
    
    if file_path.exists() and file_path.is_file():
        # Определяем MIME type для правильной отдачи файлов
        mime_type = None
        if filename.endswith('.png'):
            mime_type = 'image/png'
        elif filename.endswith('.mp4'):
            mime_type = 'video/mp4'
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            mime_type = 'image/jpeg'
        
        return send_file(file_path, mimetype=mime_type)
    else:
        logger.warning(f"Файл не найден: {file_path}")
        return f"File not found: {filename}", 404


@app.route('/api/download_results')
@require_auth()
def api_download_results():
    """API endpoint для скачивания результатов анализа в виде ZIP архива."""
    # Получаем путь к директории результатов из сессии или параметра запроса
    output_dir_path = request.args.get('dir') or session.get('last_analysis_dir')
    
    if not output_dir_path:
        return jsonify({'success': False, 'message': 'Директория результатов не найдена'}), 404
    
    results_dir = Path('results')
    analysis_dir = (results_dir / output_dir_path).resolve()
    results_dir_resolved = results_dir.resolve()
    
    # Проверяем безопасность пути
    if not str(analysis_dir).startswith(str(results_dir_resolved)):
        return jsonify({'success': False, 'message': 'Доступ запрещен'}), 403
    
    if not analysis_dir.exists() or not analysis_dir.is_dir():
        return jsonify({'success': False, 'message': 'Директория результатов не найдена'}), 404
    
    try:
        # Создаем ZIP архив в памяти
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Добавляем все файлы из директории результатов
            for file_path in analysis_dir.rglob('*'):
                if file_path.is_file():
                    # Получаем относительный путь для архива
                    arcname = file_path.relative_to(analysis_dir)
                    zip_file.write(file_path, arcname)
        
        zip_buffer.seek(0)
        
        # Генерируем имя файла с датой и временем
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f'analysis_results_{timestamp}.zip'
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
    except Exception as e:
        logger.error(f"Ошибка создания ZIP архива: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Ошибка создания архива: {str(e)}'}), 500


# ==================== АДМИН-ПАНЕЛЬ ====================

@app.route('/admin')
@require_admin()
def admin_dashboard():
    """Главная страница админ-панели."""
    return render_template('admin/dashboard.html')


@app.route('/admin/users')
@require_admin()
def admin_users():
    """Управление пользователями."""
    users = auth_manager.db.get_all_users()
    admin_logger.info(f"Администратор {session.get('email')} просматривает список пользователей")
    return render_template('admin/users.html', users=users)


@app.route('/admin/users/<int:user_id>/edit', methods=['GET', 'POST'])
@require_admin()
def admin_edit_user(user_id):
    """Редактирование пользователя."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        role = request.form.get('role', 'user')
        is_active = request.form.get('is_active') == 'on'
        password = request.form.get('password', '').strip()
        
        success, message = auth_manager.db.update_user(
            user_id=user_id,
            username=username if username else None,
            email=email if email else None,
            full_name=full_name if full_name else None,
            role=role,
            is_active=is_active,
            password=password if password else None
        )
        
        if success:
            admin_logger.info(f"Администратор {session.get('email')} обновил пользователя {user_id}")
            flash('Пользователь успешно обновлен', 'success')
        else:
            flash(f'Ошибка: {message}', 'error')
        
        return redirect(url_for('admin_users'))
    
    user = auth_manager.db.get_user_by_id(user_id)
    if not user:
        flash('Пользователь не найден', 'error')
        return redirect(url_for('admin_users'))
    
    return render_template('admin/edit_user.html', user=user)


@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@require_admin()
def admin_delete_user(user_id):
    """Удаление пользователя."""
    success, message = auth_manager.db.delete_user(user_id)
    
    if success:
        admin_logger.info(f"Администратор {session.get('email')} удалил пользователя {user_id}")
        flash('Пользователь успешно удален', 'success')
    else:
        flash(f'Ошибка: {message}', 'error')
    
    return redirect(url_for('admin_users'))


@app.route('/admin/logs')
@require_admin()
def admin_logs():
    """Просмотр логов."""
    log_type = request.args.get('type', 'all')
    lines = int(request.args.get('lines', 100))
    
    logs = []
    if log_type == 'all' or log_type == 'app':
        logs.extend(get_log_entries(LOG_FILE, lines))
    if log_type == 'all' or log_type == 'errors':
        logs.extend(get_log_entries(ERROR_LOG_FILE, lines))
    if log_type == 'all' or log_type == 'admin':
        logs.extend(get_log_entries(ADMIN_LOG_FILE, lines))
    
    # Сортируем по времени (новые сверху)
    logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    admin_logger.info(f"Администратор {session.get('email')} просматривает логи (тип: {log_type})")
    
    return render_template('admin/logs.html', logs=logs[:lines], log_type=log_type)


if __name__ == '__main__':
    # Запускаем Flask приложение
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=True)
