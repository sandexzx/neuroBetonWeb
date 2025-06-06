# Устанавливаем политику выполнения для текущего процесса
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Глобальные переменные для отслеживания процессов
$BackendProcess = $null
$FrontendProcess = $null

# Функция для красивого логотипа
function Show-Logo {
    Clear-Host
    Write-Host ""
    Write-Host "███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ██████╗ ███████╗████████╗ ██████╗ ███╗   ██╗" -ForegroundColor Cyan
    Write-Host "████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔══██╗██╔════╝╚══██╔══╝██╔═══██╗████╗  ██║" -ForegroundColor Cyan
    Write-Host "██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██████╔╝█████╗     ██║   ██║   ██║██╔██╗ ██║" -ForegroundColor Cyan
    Write-Host "██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══██╗██╔══╝     ██║   ██║   ██║██║╚██╗██║" -ForegroundColor Cyan
    Write-Host "██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝██████╔╝███████╗   ██║   ╚██████╔╝██║ ╚████║" -ForegroundColor Cyan
    Write-Host "╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🚀 Запуск системы анализа прочности бетона" -ForegroundColor Magenta
    Write-Host ""
}

# Функция проверки зависимостей
function Test-Dependencies {
    Write-Host "🔍 Проверяем зависимости..." -ForegroundColor Blue
    
    # Проверяем Python
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Python не найден"
        }
        Write-Host "✅ Python найден: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Python не найден! Установите Python с python.org" -ForegroundColor Red
        return $false
    }
    
    # Проверяем Node.js
    try {
        $nodeVersion = node --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Node.js не найден"
        }
        Write-Host "✅ Node.js найден: $nodeVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Node.js не найден! Установите Node.js с nodejs.org" -ForegroundColor Red
        return $false
    }
    
    # Проверяем npm
    try {
        $npmVersion = npm --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "npm не найден"
        }
        Write-Host "✅ npm найден: v$npmVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ npm не найден! Переустановите Node.js" -ForegroundColor Red
        return $false
    }
    
    Write-Host ""
    return $true
}

# Функция создания виртуального окружения
function Initialize-VirtualEnvironment {
    if (-not (Test-Path ".venv")) {
        Write-Host "📦 Создаем виртуальное окружение..." -ForegroundColor Yellow
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Ошибка создания виртуального окружения" -ForegroundColor Red
            return $false
        }
        Write-Host "✅ Виртуальное окружение создано" -ForegroundColor Green
    }
    return $true
}

# Функция установки зависимостей бэкенда
function Install-BackendDependencies {
    Write-Host "📦 Устанавливаем зависимости бэкенда..." -ForegroundColor Blue
    
    if (-not (Test-Path "backend\requirements.txt")) {
        Write-Host "❌ Файл requirements.txt не найден в папке backend" -ForegroundColor Red
        return $false
    }
    
    # Активируем виртуальное окружение и устанавливаем зависимости
    & ".venv\Scripts\Activate.ps1"
    pip install -r backend\requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Ошибка установки зависимостей бэкенда" -ForegroundColor Red
        return $false
    }
    
    deactivate
    Write-Host "✅ Зависимости бэкенда установлены" -ForegroundColor Green
    return $true
}

# Функция установки зависимостей фронтенда
function Install-FrontendDependencies {
    Write-Host "📦 Устанавливаем зависимости фронтенда..." -ForegroundColor Blue
    
    Push-Location frontend
    try {
        if (-not (Test-Path "package.json")) {
            Write-Host "❌ Файл package.json не найден в папке frontend" -ForegroundColor Red
            return $false
        }
        
        npm install
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Ошибка установки зависимостей фронтенда" -ForegroundColor Red
            return $false
        }
        
        Write-Host "✅ Зависимости фронтенда установлены" -ForegroundColor Green
        return $true
    }
    finally {
        Pop-Location
    }
}

# Функция для запуска бэкенда
function Start-Backend {
    Write-Host "🚀 Запускаем бэкенд сервер..." -ForegroundColor Blue
    
    $backendPath = Join-Path (Get-Location) "backend"
    $venvActivate = Join-Path (Get-Location) ".venv\Scripts\Activate.ps1"
    
    # Создаем скрипт для запуска бэкенда
    $backendScript = @"
Set-Location '$backendPath'
& '$venvActivate'
uvicorn main:app --reload --host 0.0.0.0 --port 8000
"@
    
    # Запускаем бэкенд в отдельном процессе
    $global:BackendProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendScript -PassThru -WindowStyle Normal
    
    Write-Host "✅ Бэкенд запущен (PID: $($BackendProcess.Id))" -ForegroundColor Green
    return $BackendProcess
}

# Функция для запуска фронтенда
function Start-Frontend {
    Write-Host "🚀 Запускаем фронтенд сервер..." -ForegroundColor Blue
    
    $frontendPath = Join-Path (Get-Location) "frontend"
    
    # Создаем скрипт для запуска фронтенда
    $frontendScript = @"
Set-Location '$frontendPath'
npm run dev
"@
    
    # Запускаем фронтенд в отдельном процессе
    $global:FrontendProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendScript -PassThru -WindowStyle Normal
    
    Write-Host "✅ Фронтенд запущен (PID: $($FrontendProcess.Id))" -ForegroundColor Green
    return $FrontendProcess
}

# Функция проверки готовности серверов
function Test-ServersReady {
    Write-Host "⏳ Ждем запуска серверов..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    # Проверяем бэкенд
    Write-Host "🔍 Проверяем бэкенд..." -ForegroundColor Blue
    for ($i = 1; $i -le 10; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000" -TimeoutSec 5 -ErrorAction Stop
            Write-Host "✅ Бэкенд готов!" -ForegroundColor Green
            break
        }
        catch {
            Write-Host "⏳ Ждем бэкенд... ($i/10)" -ForegroundColor Yellow
            Start-Sleep -Seconds 1
        }
    }
    
    # Проверяем фронтенд
    Write-Host "🔍 Проверяем фронтенд..." -ForegroundColor Blue
    for ($i = 1; $i -le 15; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 5 -ErrorAction Stop
            Write-Host "✅ Фронтенд готов!" -ForegroundColor Green
            break
        }
        catch {
            Write-Host "⏳ Ждем фронтенд... ($i/15)" -ForegroundColor Yellow
            Start-Sleep -Seconds 1
        }
    }
}

# Функция очистки процессов
function Stop-Servers {
    Write-Host ""
    Write-Host "🛑 Останавливаем серверы..." -ForegroundColor Yellow
    
    if ($global:BackendProcess -and !$global:BackendProcess.HasExited) {
        try {
            Stop-Process -Id $global:BackendProcess.Id -Force
            Write-Host "✅ Бэкенд остановлен" -ForegroundColor Green
        }
        catch {
            Write-Host "⚠️  Не удалось остановить бэкенд" -ForegroundColor Yellow
        }
    }
    
    if ($global:FrontendProcess -and !$global:FrontendProcess.HasExited) {
        try {
            Stop-Process -Id $global:FrontendProcess.Id -Force
            Write-Host "✅ Фронтенд остановлен" -ForegroundColor Green
        }
        catch {
            Write-Host "⚠️  Не удалось остановить фронтенд" -ForegroundColor Yellow
        }
    }
}

# Функция показа справки
function Show-Help {
    Write-Host ""
    Write-Host "Использование:" -ForegroundColor Cyan
    Write-Host "  .\start.ps1                 - Запустить приложение"
    Write-Host "  .\start.ps1 -InstallDeps    - Установить зависимости"
    Write-Host "  .\start.ps1 -Help           - Показать эту справку"
    Write-Host ""
}

# Основная функция
function Main {
    param(
        [switch]$InstallDeps,
        [switch]$Help
    )
    
    # Регистрируем обработчик для корректного завершения
    Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Stop-Servers }
    
    # Показываем логотип
    Show-Logo
    
    # Обрабатываем аргументы
    if ($Help) {
        Show-Help
        return
    }
    
    if ($InstallDeps) {
        if (-not (Test-Dependencies)) { return }
        if (-not (Initialize-VirtualEnvironment)) { return }
        if (-not (Install-BackendDependencies)) { return }
        if (-not (Install-FrontendDependencies)) { return }
        
        Write-Host ""
        Write-Host "🎉 Все зависимости установлены! Теперь запустите: .\start.ps1" -ForegroundColor Green
        return
    }
    
    # Основной процесс запуска
    if (-not (Test-Dependencies)) { return }
    
    if (-not (Test-Path ".venv")) {
        Write-Host "⚠️  Виртуальное окружение не найдено" -ForegroundColor Yellow
        Write-Host "💡 Запустите: .\start.ps1 -InstallDeps для установки зависимостей" -ForegroundColor Blue
        return
    }
    
    # Запускаем серверы
    $backendProc = Start-Backend
    Start-Sleep -Seconds 2  # Даем время бэкенду запуститься
    $frontendProc = Start-Frontend
    
    # Проверяем готовность
    Test-ServersReady
    
    # Выводим итоговую информацию
    Write-Host ""
    Write-Host "🎉 NeuroBeton успешно запущен!" -ForegroundColor Green
    Write-Host "📍 Адреса серверов:" -ForegroundColor Cyan
    Write-Host "   🌐 Фронтенд: " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:3000" -ForegroundColor Yellow
    Write-Host "   🔧 Бэкенд:   " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:8000" -ForegroundColor Yellow
    Write-Host "   📚 API Docs: " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:8000/docs" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "💡 Серверы запущены в отдельных окнах" -ForegroundColor Magenta
    Write-Host "💡 Для остановки нажмите Ctrl+C или закройте это окно" -ForegroundColor Magenta
    Write-Host ""
    
    # Предлагаем открыть браузер
    $openBrowser = Read-Host "Открыть приложение в браузере? (y/n)"
    if ($openBrowser -eq "y" -or $openBrowser -eq "Y") {
        Start-Process "http://localhost:3000"
    }
    
    # Ожидаем завершения
    try {
        Write-Host "Нажмите Ctrl+C для остановки серверов..." -ForegroundColor Gray
        while ($true) {
            Start-Sleep -Seconds 1
            # Проверяем, живы ли процессы
            if ($backendProc.HasExited -and $frontendProc.HasExited) {
                break
            }
        }
    }
    finally {
        Stop-Servers
    }
}

# Запускаем основную функцию с переданными параметрами
param(
    [switch]$InstallDeps,
    [switch]$Help
)

Main -InstallDeps:$InstallDeps -Help:$Help