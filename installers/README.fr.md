<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&pause=1000&color=31F7E4&vCenter=true&width=435&height=30&lines=ARtC+%E2%80%94+Guide+d%E2%80%99installation+(Linux+%26+Windows)" alt="Typing SVG" /></a>

---

ARtC fournit des installateurs locaux qui configurent un environnement isolé avec
Python 3.12.7 dans le répertoire du projet.
**Aucune modification n’est apportée au Python du système ni à aucune configuration globale**.

Il fonctionne aussi bien sous Linux que sous Windows 10/11.

## Index
1. Contenu généré par l’installateur
2. Installation sous Linux
3. Installation sous Windows
4. Structure attendue du projet

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&pause=1000&color=31F7E4&vCenter=true&width=435&height=30&lines=1.+Contenu+g%C3%A9n%C3%A9r%C3%A9+par+l%E2%80%99installateur" alt="Typing SVG" /></a>

---

Chaque installateur effectue les actions suivantes:

1. Télécharge une copie locale de **Python 3.12.7**
   - Linux : CPython complet compilé depuis le code source
   - Windows : distribution *embeddable* avec `import site` activé
2. Installe `pip` lorsque nécessaire
   - Requis sous Windows (via get-pip.py)
3. Crée l’environnement virtuel local `.artc`
   - Linux : `python -m venv`
   - Windows : `virtualenv` (le Python embeddable ne supporte pas `venv`)
4. Installe les dépendances depuis `requirements.txt`
5. Installe le paquet ARtC via `pip install .`
6. Supprime les fichiers temporaires utilisés durant l’installation
7. Ne modifie aucune configuration globale du système

> [!note]
> L’installation peut prendre plusieurs minutes.
> Sous Linux, Python doit être compilé depuis le code source.
> Sous Windows, la durée dépend principalement de la vitesse de téléchargement.

> [!warning]
> L’installation peut générer plusieurs centaines de Mo de données.
> En plus de l’environnement Python local, un ensemble d’audios de test est inclus.
> Tous les fichiers sont libres d’utilisation.

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&pause=1000&color=31F7E4&vCenter=true&width=435&height=30&lines=2.+Installation+sous+Linux" alt="Typing SVG" /></a>

---

## 2.1. Prérequis

Les composants suivants sont requis (outils standards pour compiler Python depuis le code source):

- gcc
- make
- tar
- curl ou wget

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
sudo pacman -S --needed base-devel curl   # ou remplacer curl par wget
```

> [!caution]
> Compiler Python peut prendre de 2 à 10 minutes selon le matériel.

## 2.2. Installation

Donnez les permissions et exécutez l’installateur:

```bash
chmod +x installers/artc_install_linux.sh
./installers/artc_install_linux.sh
```

## 2.3. Activer l’environnement virtuel

```bash
source .artc/bin/activate
```

Désactiver:

```bash
deactivate
```

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&pause=1000&color=31F7E4&vCenter=true&width=435&height=30&lines=3.+Installation+sous+Windows" alt="Typing SVG" /></a>

---

L’installateur **ne nécessite pas de droits administrateur**.
Le script utilise un *ExecutionPolicy Bypass* temporaire qui ne modifie pas le système.

## 3.1. Installation

Exécutez:

```
installers\artc_install_windows.bat
```

Cela déclenche automatiquement:

```
installers\artc_install_windows_core.ps1
```

> [!note]
> Le téléchargement du Python embeddable peut être lent en connexion instable.

## 3.2. Activer l’environnement virtuel

PowerShell:

```powershell
.\.artc\Scripts\Activate.ps1
```

CMD:

```cmd
.artc\Scripts\activate.bat
```

Désactivation dans PowerShell:

```powershell
deactivate
```

CMD:

```cmd
.artc\Scripts\deactivate.bat
```

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&pause=1000&color=31F7E4&vCenter=true&width=435&height=30&lines=4.+Structure+attendue+du+projet" alt="Typing SVG" /></a>

---

```
artc/
│
├─ installers/
│   ├─ artc_install_linux.sh
│   ├─ artc_install_windows.bat
│   └─ artc_install_windows_core.ps1
│
├─ python312/        ← Python local (auto-généré)
├─ .artc/            ← Environnement virtuel (auto-généré)
├─ src/              ← Code source
├─ test_collection/  ← Jeu d’audios de test
├─ pyproject.toml    ← Configuration du paquet
└─ requirements.txt  ← Dépendances
```
