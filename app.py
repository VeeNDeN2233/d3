

import matplotlib
matplotlib.use('Agg')

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


from auth.auth_manager import AuthManager
from core.state_manager import StateManager, AnalysisStep
from core.auth_handler import AuthHandler
from utils.logger_config import setup_logging, get_log_entries, admin_logger, LOG_FILE, ERROR_LOG_FILE, ADMIN_LOG_FILE


from flask_analysis import (
    load_models_for_flask,
    analyze_video_flask,
    get_models_status,
    _model,
    _detector,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024


Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path('templates').mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)


auth_manager = AuthManager()
_state_manager = StateManager()
_auth_handler = AuthHandler()


def require_auth():
    def decorator(f):
        def wrapper(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator


def require_admin():
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
    if 'user_id' in session:
        return redirect(url_for('main'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        if not email or not password:
            flash('Заполните все поля', 'error')
            return render_template('login.html')
        

        success, message, user_data, session_token = _auth_handler.login(email, password)
        
        if success and user_data and session_token:

            session['user_id'] = user_data.get('id')
            session['email'] = user_data.get('email')
            session['username'] = user_data.get('username')
            session['full_name'] = user_data.get('full_name')
            session['role'] = user_data.get('role', 'user')
            session['session_token'] = session_token
            

            if user_data.get('role') == 'admin':
                admin_logger.info(f"Администратор {user_data.get('email')} вошел в систему")
            else:
                logger.info(f"Пользователь {user_data.get('email')} вошел в систему")
            

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
    

    if 'user_id' in session:
        return redirect(url_for('main'))
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
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
        

        success, message, user_data, session_token = _auth_handler.register(
            email, password, password_confirm, full_name if full_name else None
        )
        
        if success and session_token and user_data:

            session['user_id'] = user_data.get('id')
            session['email'] = user_data.get('email')
            session['username'] = user_data.get('username')
            session['full_name'] = user_data.get('full_name')
            session['role'] = user_data.get('role', 'user')
            session['session_token'] = session_token
            

            logger.info(f"Новый пользователь зарегистрирован: {user_data.get('email')}")
            

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
    

    if 'user_id' in session:
        return redirect(url_for('main'))
    
    return render_template('register.html')


@app.route('/logout')
def logout():
    session_token = session.get('session_token')
    if session_token:
        auth_manager.logout(session_token)
    

    session.clear()
    

    _state_manager.update_user(is_authenticated=False, session_token=None)
    
    flash('Выход выполнен', 'info')
    return redirect(url_for('login'))


@app.route('/history')
@require_auth()
def history():
    user_id = session.get('user_id')
    results = auth_manager.db.get_user_analysis_results(user_id)
    return render_template('history.html', results=results)


@app.route('/result/<int:result_id>')
@require_auth()
def view_result(result_id):
    user_id = session.get('user_id')
    result = auth_manager.db.get_analysis_result_by_id(result_id, user_id)
    
    if not result:
        flash('Результат анализа не найден', 'error')
        return redirect(url_for('history'))
    
    return render_template('result_detail.html', result=result)


@app.route('/api/delete_result/<int:result_id>', methods=['POST'])
@require_auth()
def api_delete_result(result_id):
    user_id = session.get('user_id')
    success, message = auth_manager.db.delete_analysis_result(result_id, user_id)
    
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'message': message}), 400


@app.route('/main')
@require_auth()
def main():

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


@app.route('/api/create_patient', methods=['POST'])
@require_auth()
def api_create_patient():
    data = request.json
    last_name = data.get('last_name', '').strip()
    first_name = data.get('first_name', '').strip()
    middle_name = data.get('middle_name', '').strip() if data.get('middle_name') else None
    birth_date = data.get('birth_date', '').strip()
    
    if not last_name or not first_name or not birth_date:
        return jsonify({'success': False, 'message': 'Заполните все обязательные поля'}), 400
    
    user_id = session.get('user_id')
    success, patient_id, message = auth_manager.db.create_patient(
        user_id=user_id,
        last_name=last_name,
        first_name=first_name,
        middle_name=middle_name,
        birth_date=birth_date
    )
    
    if success:
        session['current_patient_id'] = patient_id
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'message': message
        })
    else:
        return jsonify({'success': False, 'message': message}), 400


@app.route('/api/upload_video', methods=['POST'])
@require_auth()
def api_upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'Видео не загружено'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Файл не выбран'}), 400
    
    try:

        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)
        

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
    if 'uploaded_video' not in session:
        return jsonify({'success': False, 'message': 'Видео не загружено'}), 400
    
    video_path = Path(session['uploaded_video'])
    if not video_path.exists():
        return jsonify({'success': False, 'message': 'Файл видео не найден'}), 404
    
    patient_id = request.json.get('patient_id')
    age_weeks = request.json.get('age_weeks', 12)
    gestational_age = request.json.get('gestational_age', 40)
    
    if not patient_id:
        patient_id = session.get('current_patient_id')
    
    if not patient_id:
        return jsonify({'success': False, 'message': 'Данные ребенка не заполнены'}), 400
    
    models_status = get_models_status()
    if not models_status['loaded']:
        return jsonify({
            'success': False,
            'message': 'Модели не загружены. Пожалуйста, загрузите модели сначала.',
        }), 400
    
    try:
        user_id = session.get('user_id')
        video_filename = video_path.name
        
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
        
        is_anomaly = 'аномалия' in report_text.lower() or 'риск' in report_text.lower() if report_text else False
        
        auth_manager.db.create_analysis_result(
            user_id=user_id,
            patient_id=patient_id,
            video_filename=video_filename,
            output_dir=output_dir_path,
            age_weeks=age_weeks,
            gestational_age=gestational_age,
            report_text=report_text,
            is_anomaly=is_anomaly
        )
        
        plot_url = f"/results/{plot_path}" if plot_path else None
        video_url = f"/results/{video_path_result}" if video_path_result else None
        
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
    models_status = get_models_status()
    return jsonify({
        'authenticated': True,
        'models_loaded': models_status['loaded'],
        'user_email': session.get('email'),
    })


@app.route('/results/<path:filename>')
@require_auth()
def serve_result(filename):
    results_dir = Path('results')

    file_path = (results_dir / filename).resolve()
    results_dir_resolved = results_dir.resolve()
    

    if not str(file_path).startswith(str(results_dir_resolved)):
        return "Access denied", 403
    
    if file_path.exists() and file_path.is_file():

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

    output_dir_path = request.args.get('dir') or session.get('last_analysis_dir')
    
    if not output_dir_path:
        return jsonify({'success': False, 'message': 'Директория результатов не найдена'}), 404
    
    results_dir = Path('results')
    analysis_dir = (results_dir / output_dir_path).resolve()
    results_dir_resolved = results_dir.resolve()
    

    if not str(analysis_dir).startswith(str(results_dir_resolved)):
        return jsonify({'success': False, 'message': 'Доступ запрещен'}), 403
    
    if not analysis_dir.exists() or not analysis_dir.is_dir():
        return jsonify({'success': False, 'message': 'Директория результатов не найдена'}), 404
    
    try:

        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:

            for file_path in analysis_dir.rglob('*'):
                if file_path.is_file():

                    arcname = file_path.relative_to(analysis_dir)
                    zip_file.write(file_path, arcname)
        
        zip_buffer.seek(0)
        

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




@app.route('/admin')
@require_admin()
def admin_dashboard():
    return render_template('admin/dashboard.html')


@app.route('/admin/users')
@require_admin()
def admin_users():
    search = request.args.get('search', '').strip()
    role_filter = request.args.get('role', '')
    status_filter = request.args.get('status', '')
    
    users = auth_manager.db.get_all_users()
    
    if search:
        users = [u for u in users if search.lower() in (u.get('username', '') or '').lower() 
                 or search.lower() in (u.get('email', '') or '').lower()
                 or search.lower() in (u.get('full_name', '') or '').lower()]
    
    if role_filter:
        users = [u for u in users if u.get('role') == role_filter]
    
    if status_filter:
        if status_filter == 'active':
            users = [u for u in users if u.get('is_active')]
        elif status_filter == 'inactive':
            users = [u for u in users if not u.get('is_active')]
    
    total_users = len(auth_manager.db.get_all_users())
    active_users = len([u for u in auth_manager.db.get_all_users() if u.get('is_active')])
    admin_count = len([u for u in auth_manager.db.get_all_users() if u.get('role') == 'admin'])
    
    admin_logger.info(f"Администратор {session.get('email')} просматривает список пользователей")
    return render_template('admin/users.html', 
                         users=users, 
                         search=search,
                         role_filter=role_filter,
                         status_filter=status_filter,
                         total_users=total_users,
                         active_users=active_users,
                         admin_count=admin_count)


@app.route('/admin/users/<int:user_id>/edit', methods=['GET', 'POST'])
@require_admin()
def admin_edit_user(user_id):
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


@app.route('/admin/users/create', methods=['GET', 'POST'])
@require_admin()
def admin_create_user():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        password_confirm = request.form.get('password_confirm', '').strip()
        full_name = request.form.get('full_name', '').strip()
        role = request.form.get('role', 'user')
        is_active = request.form.get('is_active') == 'on'
        
        if not username or not password:
            flash('Заполните имя пользователя и пароль', 'error')
            return render_template('admin/create_user.html')
        
        if password != password_confirm:
            flash('Пароли не совпадают', 'error')
            return render_template('admin/create_user.html')
        
        success, message = auth_manager.db.create_user(
            username=username,
            password=password,
            email=email if email else None,
            full_name=full_name if full_name else None,
            role=role
        )
        
        if success:
            if not is_active:
                user = auth_manager.db.get_user_by_username(username)
                if user:
                    auth_manager.db.update_user(
                        user_id=user['id'],
                        is_active=False
                    )
            admin_logger.info(f"Администратор {session.get('email')} создал пользователя {username}")
            flash('Пользователь успешно создан', 'success')
            return redirect(url_for('admin_users'))
        else:
            flash(f'Ошибка: {message}', 'error')
    
    return render_template('admin/create_user.html')


@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@require_admin()
def admin_delete_user(user_id):
    success, message = auth_manager.db.delete_user(user_id)
    
    if success:
        admin_logger.info(f"Администратор {session.get('email')} удалил пользователя {user_id}")
        flash('Пользователь успешно удален', 'success')
    else:
        flash(f'Ошибка: {message}', 'error')
    
    return redirect(url_for('admin_users'))


@app.route('/admin/users/bulk_action', methods=['POST'])
@require_admin()
def admin_bulk_action():
    action = request.form.get('action')
    user_ids = request.form.getlist('user_ids')
    
    if not user_ids:
        flash('Выберите хотя бы одного пользователя', 'error')
        return redirect(url_for('admin_users'))
    
    user_ids = [int(uid) for uid in user_ids]
    success_count = 0
    
    if action == 'activate':
        for user_id in user_ids:
            success, _ = auth_manager.db.update_user(user_id=user_id, is_active=True)
            if success:
                success_count += 1
        admin_logger.info(f"Администратор {session.get('email')} активировал {success_count} пользователей")
        flash(f'Активировано пользователей: {success_count}', 'success')
    
    elif action == 'deactivate':
        for user_id in user_ids:
            success, _ = auth_manager.db.update_user(user_id=user_id, is_active=False)
            if success:
                success_count += 1
        admin_logger.info(f"Администратор {session.get('email')} деактивировал {success_count} пользователей")
        flash(f'Деактивировано пользователей: {success_count}', 'success')
    
    elif action == 'delete':
        for user_id in user_ids:
            success, _ = auth_manager.db.delete_user(user_id)
            if success:
                success_count += 1
        admin_logger.info(f"Администратор {session.get('email')} удалил {success_count} пользователей")
        flash(f'Удалено пользователей: {success_count}', 'success')
    
    return redirect(url_for('admin_users'))


@app.route('/admin/logs')
@require_admin()
def admin_logs():
    log_type = request.args.get('type', 'all')
    lines = int(request.args.get('lines', 100))
    
    logs = []
    if log_type == 'all' or log_type == 'app':
        logs.extend(get_log_entries(LOG_FILE, lines))
    if log_type == 'all' or log_type == 'errors':
        logs.extend(get_log_entries(ERROR_LOG_FILE, lines))
    if log_type == 'all' or log_type == 'admin':
        logs.extend(get_log_entries(ADMIN_LOG_FILE, lines))
    

    logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    admin_logger.info(f"Администратор {session.get('email')} просматривает логи (тип: {log_type})")
    
    return render_template('admin/logs.html', logs=logs[:lines], log_type=log_type)


if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=True)
