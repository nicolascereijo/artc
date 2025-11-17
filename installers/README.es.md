# **ARtC — Guía de Instalación (Linux y Windows)**

Este proyecto incluye instaladores locales que permiten configurar un entorno de Python 3.12 completamente aislado dentro del propio directorio del proyecto, sin modificar configuraciones del sistema operativo.

El proceso es totalmente portable y funciona tanto en Linux como en Windows 10/11.

---

## Contenidos generados por el instalador

Cada instalador realiza lo siguiente:

1. Descarga una versión local de Python 3.12.x dentro del proyecto
2. Crea un entorno virtual interno llamado `.artc`
3. Instala dependencias desde `requirements.txt`
4. Instala el propio paquete del proyecto usando `pip install .`
5. Limpia archivos temporales
6. No modifica ninguna configuración global del sistema

---

# **Instalación en Linux**

## Requisitos previos

Es necesario tener instaladas estas herramientas:

- gcc
- make
- tar
- curl o wget

En distribuciones basadas en Debian/Ubuntu:

```bash
sudo apt install build-essential curl
```

En Fedora/RHEL:

```bash
sudo dnf install gcc make tar wget
```

---

## Instalación

Ejecutar el instalador para Linux:

```bash
./installers/artc-install-linux.sh
```

Si no tiene permisos de ejecución:

```bash
chmod +x installers/artc-install-linux.sh
```

---

## Activar el entorno virtual

```bash
source .artc/bin/activate
```

## Desactivar el entorno

```bash
deactivate
```

---

# **Instalación en Windows (10 y 11)**

La instalación en Windows se realiza mediante un archivo `.bat`, que ejecuta el script PowerShell interno sin necesidad de cambiar las políticas de ejecución.

No se requieren permisos de administrador.

---

## Instalación

Ejecutar:

```
installers\artc_install_windows.bat
```

Este archivo lanzará automáticamente:

```
installers\artc_install_windows_core.ps1
```

con `ExecutionPolicy Bypass` aplicado solo durante esta ejecución.

---

## Activar el entorno virtual

En PowerShell:

```powershell
.\.artc\Scripts\Activate.ps1
```

En CMD:

```cmd
.artc\Scripts\activate.bat
```

---

## Desactivar el entorno

En PowerShell:

```powershell
deactivate
```

En CMD:

```cmd
.artc\Scripts\deactivate.bat
```

---

# Estructura esperada del proyecto

```
artc/
│
├─ installers/
│   ├─ artc-install-linux.sh
│   ├─ artc_install_windows.bat
│   └─ artc_install_windows_core.ps1
│
├─ python312/        ← Python local (auto-generado)
├─ .artc/            ← Entorno virtual (auto-generado)
├─ src/              ← Código fuente
├─ setup.py          ← Instalación del paquete
└─ requirements.txt  ← Dependencias
```

---

# Notas importantes

- Ningún instalador requiere permisos root/administrador.
- No se altera el Python del sistema operativo.
- En Windows no se cambia la política de ejecución global.
- Todo el contenido generado vive dentro del proyecto y puede borrarse sin riesgo.
