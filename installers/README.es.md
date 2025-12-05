<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&pause=1000&color=31F7E4&vCenter=true&width=435&height=30&lines=ARtC+%E2%80%94+Gu%C3%ADa+de+Instalaci%C3%B3n+(Linux+y+Windows)" alt="Typing SVG" /></a>

---

ARtC proporciona instaladores locales que configuran un entorno aislado con
Python 3.12.7 dentro del propio directorio del proyecto.
**No modifica el Python del sistema ni configura variables globales**.

Funciona tanto en Linux como en Windows 10/11.

## Índice
1. Contenidos generados por el instalador
2. Instalación en Linux
3. Instalación en Windows
4. Estructura esperada del proyecto

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&pause=1000&color=31F7E4&vCenter=true&width=435&height=30&lines=1.+Contenidos+generados+por+el+instalador" alt="Typing SVG" /></a>

---

Cada instalador realiza lo siguiente:

1. Descarga una versión local de **Python 3.12.7**
   - Linux: CPython completo compilado desde código fuente
   - Windows: distribución *embeddable* con `import site` activado
2. Instala `pip` cuando es necesario
   - Obligatorio en Windows (mediante get-pip.py)
3. Crea un entorno virtual local `.artc`
   - Linux: `python -m venv`
   - Windows: `virtualenv` (el Python embebido no soporta `venv`)
4. Instala las dependencias desde `requirements.txt`
5. Instala el paquete ARtC mediante `pip install .`
6. Elimina archivos temporales de instalación
7. No modifica ninguna configuración global del sistema operativo

> [!note]
> La instalación puede tardar varios minutos.
> En Linux, Python debe compilarse desde código.
> En Windows, el tiempo depende principalmente de la velocidad de descarga.

> [!warning]
> La instalación puede crear varios cientos de MB de archivos. Aparte del Python local, se incluye un set de audios de prueba de varias fuentes. Todos ellos son de uso libre.

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&pause=1000&color=31F7E4&vCenter=true&width=435&height=30&lines=2.+Instalaci%C3%B3n+en+Linux" alt="Typing SVG" /></a>

---

## 2.1. Requisitos previos

Son necesarios los siguientes componentes (herramientas habituales de compilación para construir Python desde código fuente):

- gcc
- make
- tar
- curl o wget

Debian/Ubuntu:

```bash
sudo apt install build-essential curl
```

Fedora/RHEL:

```bash
sudo dnf install gcc make tar wget
```

Arch/Manjaro:

```bash
sudo pacman -S --needed base-devel curl   # o reemplazar curl por wget
```

> [!caution]
> Compilar Python puede tardar entre 2 y 10 minutos dependiendo del hardware.

## 2.2. Instalación

Dar permisos y ejecutar el instalador:

```bash
chmod +x installers/artc_install_linux.sh
./installers/artc_install_linux.sh
```

## 2.3. Activar el entorno virtual

```bash
source .artc/bin/activate
```

Desactivar:

```bash
deactivate
```

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&pause=1000&color=31F7E4&vCenter=true&width=435&height=30&lines=3.+Instalaci%C3%B3n+en+Windows" alt="Typing SVG" /></a>

---

El instalador **no requiere permisos de administrador**.
El script usa un *ExecutionPolicy Bypass* temporal que no modifica el sistema.

## 3.1. Instalación

Ejecutar:

```
installers\artc_install_windows.bat
```

Esto ejecuta automáticamente:

```
installers\artc_install_windows_core.ps1
```

> [!note]
> La descarga del Python embebido puede ser lenta en conexiones inestables.

## 3.2. Activar el entorno virtual

PowerShell:

```powershell
.\.artc\Scripts\Activate.ps1
```

CMD:

```cmd
.artc\Scripts\activate.bat
```

Desactivar en PowerShell:

```powershell
deactivate
```

CMD:

```cmd
.artc\Scripts\deactivate.bat
```

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&pause=1000&color=31F7E4&vCenter=true&width=435&height=30&lines=4.+Estructura+esperada+del+proyecto" alt="Typing SVG" /></a>

---

```
artc/
│
├─ installers/
│   ├─ artc_install_linux.sh
│   ├─ artc_install_windows.bat
│   └─ artc_install_windows_core.ps1
│
├─ python312/        ← Python local (auto-generado)
├─ .artc/            ← Entorno virtual (auto-generado)
├─ src/              ← Código fuente
├─ test_collection/  ← Conjunto de audios de prueba
├─ pyproject.toml    ← Configuración del paquete
└─ requirements.txt  ← Dependencias
```
