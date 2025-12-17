// JavaScript для страницы анализа

let currentStep = 0;
let uploadedVideo = null;
let analysisInProgress = false;
let patientId = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeAnalysisPage();
});

function initializeAnalysisPage() {
    // Инициализация шагов
    setupStepNavigation();
    
    // Инициализация загрузки файлов
    setupFileUpload();
    
    // Автоматическая загрузка моделей при загрузке страницы
    autoLoadModels();
    
    // Инициализация анализа
    setupAnalysis();
}

function setupStepNavigation() {
    // Переход от шага 0 к шагу 1 (данные ребенка -> загрузка видео)
    document.getElementById('next-to-step1')?.addEventListener('click', function() {
        savePatientData();
    });
    
    // Переход к шагу 2
    document.getElementById('next-to-step2')?.addEventListener('click', function() {
        goToStep(2);
    });
    
    // Переход к шагу 3
    document.getElementById('next-to-step3')?.addEventListener('click', function() {
        goToStep(3);
    });
    
    // Назад к шагу 0
    document.getElementById('back-to-step0')?.addEventListener('click', function() {
        goToStep(0);
    });
    
    // Назад к шагу 1
    document.getElementById('back-to-step1')?.addEventListener('click', function() {
        goToStep(1);
    });
    
    // Назад к шагу 2
    document.getElementById('back-to-step2')?.addEventListener('click', function() {
        goToStep(2);
    });
    
    // Новый анализ
    document.getElementById('new-analysis')?.addEventListener('click', function() {
        resetAnalysis();
        goToStep(0);
    });
}

function savePatientData() {
    const lastName = document.getElementById('patient_last_name').value.trim();
    const firstName = document.getElementById('patient_first_name').value.trim();
    const middleName = document.getElementById('patient_middle_name').value.trim();
    const birthDate = document.getElementById('patient_birth_date').value;
    
    if (!lastName || !firstName || !birthDate) {
        alert('Пожалуйста, заполните все обязательные поля (Фамилия, Имя, Дата рождения)');
        return;
    }
    
    fetch('/api/create_patient', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify({
            last_name: lastName,
            first_name: firstName,
            middle_name: middleName || null,
            birth_date: birthDate
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            patientId = data.patient_id;
            goToStep(1);
        } else {
            alert('Ошибка сохранения данных ребенка: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Ошибка сохранения данных ребенка:', error);
        alert('Ошибка сохранения данных ребенка');
    });
}

function goToStep(step) {
    // Скрываем все панели
    for (let i = 0; i <= 4; i++) {
        const panel = document.getElementById(`step${i}-panel`);
        const stepEl = document.getElementById(`step-${i}`);
        
        if (panel) {
            panel.style.display = i === step ? 'block' : 'none';
            panel.classList.toggle('active', i === step);
        }
        
        if (stepEl) {
            stepEl.classList.toggle('active', i === step);
            if (i < step) {
                stepEl.classList.add('completed');
            } else {
                stepEl.classList.remove('completed');
            }
        }
    }
    
    currentStep = step;
}

function setupFileUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('video-input');
    const nextBtn = document.getElementById('next-to-step2');
    
    if (!uploadArea || !fileInput) return;
    
    // Клик по области загрузки
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Перетаскивание файла
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // Выбор файла
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    if (!file.type.startsWith('video/')) {
        alert('Пожалуйста, выберите видео файл');
        return;
    }
    
    uploadedVideo = file;
    
    // Обновляем UI
    const uploadArea = document.getElementById('upload-area');
    const uploadStatus = document.getElementById('upload-status');
    const nextBtn = document.getElementById('next-to-step2');
    
    if (uploadArea) {
        uploadArea.classList.add('has-file');
        uploadArea.querySelector('.upload-placeholder').innerHTML = `
            <p><strong>${file.name}</strong></p>
            <p class="upload-hint">Размер: ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
        `;
    }
    
    if (uploadStatus) {
        uploadStatus.style.display = 'block';
    }
    
    if (nextBtn) {
        nextBtn.disabled = false;
    }
    
    // Загружаем файл на сервер
    uploadVideoToServer(file);
}

function uploadVideoToServer(file) {
    const formData = new FormData();
    formData.append('video', file);
    
    fetch('/api/upload_video', {
        method: 'POST',
        body: formData,
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Видео загружено успешно');
        } else {
            alert('Ошибка загрузки видео: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Ошибка загрузки видео:', error);
        alert('Ошибка загрузки видео');
    });
}

function autoLoadModels() {
    // Проверяем, нужно ли загружать модели
    const loadingBanner = document.getElementById('model-loading-banner');
    if (!loadingBanner) return;
    
    // Проверяем статус загрузки моделей через API
    fetch('/api/check_status')
        .then(response => response.json())
        .then(data => {
            if (data.models_loaded) {
                // Модели уже загружены, баннер остается скрытым
                console.log('Модели уже загружены');
                return;
            }
            
            // Показываем баннер и начинаем загрузку
            loadingBanner.style.display = 'block';
            console.log('Начало автоматической загрузки моделей...');
            loadModelsWithProgress();
        })
        .catch(error => {
            console.error('Ошибка проверки статуса:', error);
            // В случае ошибки все равно показываем баннер
            loadingBanner.style.display = 'block';
            loadModelsWithProgress();
        });
}

function hideModelLoadingBanner() {
    const loadingBanner = document.getElementById('model-loading-banner');
    if (loadingBanner) {
        // Добавляем класс для анимации исчезновения
        loadingBanner.classList.add('fade-out');
        
        // Удаляем элемент после завершения анимации
        setTimeout(() => {
            loadingBanner.style.display = 'none';
        }, 500); // Время анимации
    }
}

function loadModelsWithProgress() {
    const progressFill = document.getElementById('model-progress-fill');
    const progressText = document.getElementById('model-progress-text');
    const loadingBanner = document.getElementById('model-loading-banner');
    
    // Этапы загрузки
    const loadingSteps = [
        { id: 'loading-step-1', text: 'Проверка GPU...', progress: 10 },
        { id: 'loading-step-2', text: 'Загрузка конфигурации...', progress: 25 },
        { id: 'loading-step-3', text: 'Загрузка модели LSTM...', progress: 50 },
        { id: 'loading-step-4', text: 'Загрузка детектора аномалий...', progress: 75 },
        { id: 'loading-step-5', text: 'Инициализация процессоров...', progress: 90 },
    ];
    
    let currentStep = 0;
    
    // Функция обновления прогресса
    function updateProgress(stepIndex, progress) {
        if (progressFill) {
            progressFill.style.width = progress + '%';
        }
        
        if (progressText) {
            if (stepIndex < loadingSteps.length) {
                progressText.textContent = loadingSteps[stepIndex].text;
            } else {
                progressText.textContent = 'Завершение загрузки...';
            }
        }
        
        // Обновляем визуальные индикаторы шагов
        loadingSteps.forEach((step, idx) => {
            const stepElement = document.getElementById(step.id);
            if (stepElement) {
                stepElement.classList.remove('active', 'completed');
                if (idx < stepIndex) {
                    stepElement.classList.add('completed');
                    const icon = stepElement.querySelector('.step-icon');
                    if (icon) icon.textContent = '✓';
                } else if (idx === stepIndex) {
                    stepElement.classList.add('active');
                }
            }
        });
    }
    
    // Симулируем прогресс загрузки
    function simulateProgress() {
        if (currentStep < loadingSteps.length) {
            const step = loadingSteps[currentStep];
            updateProgress(currentStep, step.progress);
            currentStep++;
            
            // Задержка между шагами (симуляция)
            setTimeout(simulateProgress, 800);
        } else {
            // Завершаем прогресс и отправляем запрос
            updateProgress(currentStep, 95);
            performModelLoad();
        }
    }
    
    // Фактическая загрузка моделей
    function performModelLoad() {
        fetch('/api/load_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(response => response.json())
        .then(data => {
            // Завершаем прогресс
            if (progressFill) {
                progressFill.style.width = '100%';
            }
            
            // Отмечаем все шаги как завершенные
            loadingSteps.forEach(step => {
                const stepElement = document.getElementById(step.id);
                if (stepElement) {
                    stepElement.classList.add('completed');
                    const icon = stepElement.querySelector('.step-icon');
                    if (icon) icon.textContent = '✓';
                }
            });
            
            if (data.success) {
                if (progressText) {
                    progressText.textContent = 'Модели загружены успешно!';
                }
                
                // Показываем успешное сообщение и скрываем баннер через задержку
                setTimeout(() => {
                    // Разблокируем кнопку запуска анализа
                    checkAnalysisReady();
                    
                    // Скрываем баннер с анимацией
                    setTimeout(() => {
                        hideModelLoadingBanner();
                    }, 1500); // Показываем успешное сообщение 1.5 секунды
                }, 500);
            } else {
                if (progressText) {
                    progressText.textContent = 'Ошибка загрузки: ' + (data.message || 'Неизвестная ошибка');
                }
                
                // Показываем ошибку (баннер остается видимым)
                const loadingContainer = document.getElementById('model-loading-container');
                if (loadingContainer) {
                    setTimeout(() => {
                        loadingContainer.innerHTML = `
                            <div class="status-error" style="margin: 0;">
                                <strong>Ошибка загрузки моделей</strong><br>
                                ${data.message || 'Неизвестная ошибка'}<br>
                                <button class="btn-primary" onclick="location.reload()" style="margin-top: 12px;">
                                    Попробовать снова
                                </button>
                            </div>
                        `;
                    }, 1000);
                }
            }
        })
        .catch(error => {
            console.error('Ошибка загрузки моделей:', error);
            if (progressText) {
                progressText.textContent = 'Ошибка соединения с сервером';
            }
            
            const loadingContainer = document.getElementById('model-loading-container');
            if (loadingContainer) {
                setTimeout(() => {
                    loadingContainer.innerHTML = `
                        <div class="status-error" style="margin: 0;">
                            <strong>Ошибка соединения</strong><br>
                            Не удалось подключиться к серверу<br>
                            <button class="btn-primary" onclick="location.reload()" style="margin-top: 12px;">
                                Обновить страницу
                            </button>
                        </div>
                    `;
                }, 1000);
            }
        });
    }
    
    // Начинаем симуляцию прогресса
    simulateProgress();
}

function setupAnalysis() {
    const startBtn = document.getElementById('start-analysis');
    const cancelBtn = document.getElementById('cancel-analysis');
    
    if (startBtn) {
        startBtn.addEventListener('click', function() {
            startAnalysis();
        });
    }
    
    if (cancelBtn) {
        cancelBtn.addEventListener('click', function() {
            cancelAnalysis();
        });
    }
}

function startAnalysis() {
    if (analysisInProgress) return;
    
    const ageWeeks = parseInt(document.getElementById('age_weeks').value) || 12;
    const gestationalAge = parseInt(document.getElementById('gestational_age').value) || 40;
    
    analysisInProgress = true;
    
    // Переходим к шагу 3
    goToStep(3);
    
    // Показываем прогресс
    const progressDiv = document.getElementById('analysis-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const startBtn = document.getElementById('start-analysis');
    const cancelBtn = document.getElementById('cancel-analysis');
    
    if (progressDiv) progressDiv.style.display = 'block';
    if (startBtn) {
        startBtn.disabled = true;
        // Изменяем текст кнопки (второй span с текстом)
        const btnSpans = startBtn.querySelectorAll('span');
        if (btnSpans.length >= 2) {
            // Если есть иконка и текст - меняем второй span
            btnSpans[1].textContent = 'Анализ запущен';
        } else if (btnSpans.length === 1) {
            // Если только один span
            btnSpans[0].textContent = 'Анализ запущен';
        } else {
            // Если нет span элементов
            startBtn.textContent = 'Анализ запущен';
        }
    }
    if (cancelBtn) cancelBtn.style.display = 'inline-block';
    
    // Симуляция прогресса (в реальности будет через WebSocket или polling)
    let progress = 0;
    const progressInterval = setInterval(function() {
        progress += 10;
        if (progressFill) progressFill.style.width = progress + '%';
        if (progressText) {
            const steps = [
                'Инициализация анализа...',
                'Обработка видео...',
                'Извлечение ключевых точек...',
                'Анализ движений...',
                'Генерация отчета...',
                'Завершение...'
            ];
            const stepIndex = Math.floor(progress / 20);
            progressText.textContent = steps[Math.min(stepIndex, steps.length - 1)];
        }
        
        if (progress >= 100) {
            clearInterval(progressInterval);
        }
    }, 500);
    
    // Запускаем анализ
    fetch('/api/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify({
            patient_id: patientId,
            age_weeks: ageWeeks,
            gestational_age: gestationalAge
        })
    })
    .then(response => response.json())
    .then(data => {
        clearInterval(progressInterval);
        analysisInProgress = false;
        
        if (data.success) {
            // Переходим к шагу 4 с результатами
            goToStep(4);
            
            // Отображаем результаты
            displayResults(data);
        } else {
            alert('Ошибка анализа: ' + data.message);
            if (startBtn) {
                startBtn.disabled = false;
                // Возвращаем исходный текст кнопки
                startBtn.innerHTML = '<span class="btn-icon">▶</span><span>Запустить анализ движений</span>';
            }
            if (cancelBtn) cancelBtn.style.display = 'none';
        }
    })
    .catch(error => {
        clearInterval(progressInterval);
        analysisInProgress = false;
        console.error('Ошибка анализа:', error);
        alert('Ошибка анализа');
        if (startBtn) {
            startBtn.disabled = false;
            // Возвращаем исходный текст кнопки
            startBtn.innerHTML = '<span class="btn-icon">▶</span><span>Запустить анализ движений</span>';
        }
        if (cancelBtn) cancelBtn.style.display = 'none';
    });
}

function displayResults(data) {
    // Отображаем отчет
    const reportContent = document.getElementById('report-content');
    if (reportContent && data.report_text) {
        reportContent.textContent = data.report_text;
    }
    
    // Отображаем видео
    const skeletonVideo = document.getElementById('skeleton-video');
    if (skeletonVideo && data.video_path) {
        skeletonVideo.src = data.video_path;
    }
    
    // Отображаем график
    const anomalyPlot = document.getElementById('anomaly-plot');
    if (anomalyPlot && data.plot_path) {
        anomalyPlot.src = data.plot_path;
    }
    
    // Показываем кнопку скачивания, если есть директория результатов
    const downloadBtn = document.getElementById('download-results');
    if (downloadBtn && data.output_dir) {
        downloadBtn.style.display = 'inline-block';
        downloadBtn.onclick = function() {
            downloadResults(data.output_dir);
        };
    }
}

function downloadResults(outputDir) {
    // Скачиваем ZIP архив с результатами
    const url = `/api/download_results?dir=${encodeURIComponent(outputDir)}`;
    const link = document.createElement('a');
    link.href = url;
    link.download = 'analysis_results.zip';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function cancelAnalysis() {
    analysisInProgress = false;
    // Здесь можно добавить логику отмены через API
    alert('Анализ отменен');
}

function resetAnalysis() {
    currentStep = 0;
    uploadedVideo = null;
    analysisInProgress = false;
    patientId = null;

    // Скрываем кнопку скачивания
    const downloadBtn = document.getElementById('download-results');
    if (downloadBtn) {
        downloadBtn.style.display = 'none';
    }

    // Сбрасываем форму данных ребенка
    document.getElementById('patient_last_name').value = '';
    document.getElementById('patient_first_name').value = '';
    document.getElementById('patient_middle_name').value = '';
    document.getElementById('patient_birth_date').value = '';

    // Сбрасываем форму загрузки видео
    const fileInput = document.getElementById('video-input');
    const uploadArea = document.getElementById('upload-area');
    const nextBtn = document.getElementById('next-to-step2');
    
    if (fileInput) fileInput.value = '';
    if (uploadArea) {
        uploadArea.classList.remove('has-file');
        uploadArea.querySelector('.upload-placeholder').innerHTML = `
            <p>Перетащите видео сюда или нажмите для выбора</p>
            <p class="upload-hint">Формат: MP4, AVI, MOV, MKV, WebM</p>
        `;
    }
    if (nextBtn) nextBtn.disabled = true;
}

// Проверка готовности к анализу
let lastCheckTime = 0;
const CHECK_INTERVAL = 2000; // Проверяем каждые 2 секунды вместо 500ms

function checkAnalysisReady() {
    const now = Date.now();
    // Ограничиваем частоту проверок
    if (now - lastCheckTime < CHECK_INTERVAL) {
        return;
    }
    lastCheckTime = now;
    
    // Проверяем статус через API
    fetch('/api/check_status')
        .then(response => response.json())
        .then(data => {
            const startBtn = document.getElementById('start-analysis');
            
            if (startBtn && uploadedVideo && data.models_loaded) {
                startBtn.disabled = false;
            } else if (startBtn) {
                startBtn.disabled = true;
            }
        })
        .catch(error => {
            console.error('Ошибка проверки статуса:', error);
        });
}

// Проверяем готовность при изменении состояния (реже)
setInterval(checkAnalysisReady, CHECK_INTERVAL);
