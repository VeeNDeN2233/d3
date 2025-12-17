"""
Контроллер DOM для агрессивного управления элементами интерфейса.
Использует JavaScript для полного удаления элементов из DOM.
"""

import logging

logger = logging.getLogger(__name__)


def get_dom_controller_js() -> str:
    """
    Получить JavaScript код для агрессивного управления DOM.
    
    Returns:
        JavaScript код для встраивания в HTML
    """
    return """
    <script>
    (function() {
        'use strict';
        
        // Класс для управления DOM элементами
        class DOMController {
            constructor() {
                this.removedElements = new Set();
                this.observer = null;
                this.checkInterval = null;
                this.init();
            }
            
            init() {
                // Немедленная проверка
                this.checkAndRemoveLoginElements();
                
                // Проверка при загрузке
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', () => this.checkAndRemoveLoginElements());
                }
                
                // Постоянный мониторинг изменений DOM
                this.startObserver();
                
                // Периодическая проверка
                this.startPeriodicCheck();
            }
            
            checkAndRemoveLoginElements() {
                // Проверяем наличие header (признак авторизации)
                const headerPanel = document.querySelector('.header-panel');
                const hasHeader = headerPanel && 
                                 headerPanel.offsetParent !== null && 
                                 (headerPanel.textContent.includes('@') || 
                                  Array.from(headerPanel.querySelectorAll('span')).some(span => 
                                      span.textContent && span.textContent.includes('@')
                                  ));
                
                // Также проверяем наличие кнопки "Выйти" как признак авторизации
                const logoutButton = document.querySelector('button:contains("Выйти")') ||
                                   Array.from(document.querySelectorAll('button')).find(btn => 
                                       btn.textContent && btn.textContent.includes('Выйти')
                                   );
                
                if (hasHeader || logoutButton) {
                    this.removeLoginElements();
                }
            }
            
            removeLoginElements() {
                // Находим все кнопки "Войти в систему"
                const buttons = Array.from(document.querySelectorAll('button'));
                let removedCount = 0;
                buttons.forEach(btn => {
                    const text = btn.textContent || btn.innerText || '';
                    if (text.includes('Войти в систему') && !this.removedElements.has(btn)) {
                        // Полностью удаляем из DOM
                        btn.remove();
                        this.removedElements.add(btn);
                        removedCount++;
                    }
                });
                if (removedCount > 0) {
                    console.log(`✅ Удалено ${removedCount} кнопок "Войти в систему"`);
                }
                
                // Находим страницу входа по наличию полей пароля
                const passwordFields = document.querySelectorAll('input[type="password"]');
                passwordFields.forEach(field => {
                    const column = field.closest('.gr-column');
                    if (column) {
                        // Проверяем, что это страница входа (есть поле email или текст "Email")
                        const hasEmailField = column.querySelector('input[type="text"][placeholder*="email" i]') ||
                                            column.querySelector('input[type="email"]') ||
                                            column.textContent.includes('Email') ||
                                            column.textContent.includes('email');
                        
                        if (hasEmailField && !this.removedElements.has(column)) {
                            // Полностью удаляем из DOM
                            column.remove();
                            this.removedElements.add(column);
                            console.log('✅ Удалена страница входа (по полю пароля)');
                        }
                    }
                });
                
                // Также удаляем через селектор по классу или data-атрибуту
                const loginColumns = document.querySelectorAll('.gr-column');
                loginColumns.forEach(col => {
                    if (this.removedElements.has(col)) return;
                    if (col.closest('.header-panel')) return; // Не трогаем header
                    
                    // Проверяем наличие элементов страницы входа
                    const buttons = col.querySelectorAll('button');
                    let hasLoginButton = false;
                    buttons.forEach(btn => {
                        const text = btn.textContent || btn.innerText || '';
                        if (text.includes('Войти в систему')) {
                            hasLoginButton = true;
                        }
                    });
                    
                    const hasPasswordField = col.querySelector('input[type="password"]');
                    const emailInputs = Array.from(col.querySelectorAll('input'));
                    const hasEmailField = col.querySelector('input[type="email"]') || 
                                        emailInputs.some(inp => {
                                            const placeholder = (inp.placeholder || '').toLowerCase();
                                            const label = (inp.getAttribute('aria-label') || '').toLowerCase();
                                            return placeholder.includes('email') || 
                                                   label.includes('email') ||
                                                   placeholder.includes('почт');
                                        });
                    
                    // Проверяем наличие текста "Email" или "Вход"
                    const hasLoginText = col.textContent.includes('Email') || 
                                       col.textContent.includes('Вход') ||
                                       col.textContent.includes('Доступ к системе');
                    
                    if ((hasLoginButton || (hasPasswordField && (hasEmailField || hasLoginText))) && 
                        !this.removedElements.has(col)) {
                        col.remove();
                        this.removedElements.add(col);
                        console.log('✅ Удалена страница входа (по селектору)');
                    }
                });
            }
            
            startObserver() {
                // MutationObserver для отслеживания изменений DOM
                this.observer = new MutationObserver((mutations) => {
                    let shouldCheck = false;
                    mutations.forEach(mutation => {
                        if (mutation.addedNodes.length > 0) {
                            shouldCheck = true;
                        }
                    });
                    
                    if (shouldCheck) {
                        setTimeout(() => this.checkAndRemoveLoginElements(), 100);
                    }
                });
                
                this.observer.observe(document.body, {
                    childList: true,
                    subtree: true,
                    attributes: false
                });
            }
            
            startPeriodicCheck() {
                // Проверка каждые 100мс для более быстрой реакции
                this.checkInterval = setInterval(() => {
                    this.checkAndRemoveLoginElements();
                }, 100);
            }
            
            destroy() {
                if (this.observer) {
                    this.observer.disconnect();
                }
                if (this.checkInterval) {
                    clearInterval(this.checkInterval);
                }
            }
        }
        
        // Создаем глобальный экземпляр контроллера
        window.domController = new DOMController();
        
        // Также экспортируем для отладки
        console.log('DOM Controller инициализирован');
    })();
    </script>
    """
