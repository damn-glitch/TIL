#!/usr/bin/env python3
"""
TIL IDE - Integrated Development Environment
Author: Alisher Beisembekov
"Simpler than Python. Faster than C. Smarter than all."
"""

import sys
import os
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeView, QTextEdit, QPlainTextEdit, QTabWidget,
    QToolBar, QStatusBar, QFileDialog, QMessageBox, QLabel,
    QLineEdit, QPushButton, QFrame, QMenuBar, QMenu,
    QDialog, QDialogButtonBox, QFormLayout, QComboBox, QListWidget,
    QListWidgetItem
)
from PyQt6.QtCore import Qt, QDir, QTimer, QSize
from PyQt6.QtGui import (
    QFont, QColor, QTextCharFormat, QSyntaxHighlighter, QAction,
    QKeySequence, QIcon, QPalette, QTextCursor, QPainter, QFontMetrics
)


# ═══════════════════════════════════════════════════════════════════════════════
#                              SYNTAX HIGHLIGHTER
# ═══════════════════════════════════════════════════════════════════════════════

class TILHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for TIL language"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#ff79c6"))  # Pink
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            'fn', 'let', 'var', 'const', 'if', 'else', 'elif', 'for', 'while',
            'loop', 'break', 'continue', 'return', 'struct', 'enum', 'impl',
            'trait', 'pub', 'use', 'import', 'from', 'as', 'in', 'match',
            'type', 'and', 'or', 'not', 'true', 'false', 'nil', 'self'
        ]
        for word in keywords:
            self.highlighting_rules.append((
                f'\\b{word}\\b', keyword_format
            ))

        # Types
        type_format = QTextCharFormat()
        type_format.setForeground(QColor("#8be9fd"))  # Cyan
        types = ['int', 'float', 'str', 'bool', 'char', 'void', 'any']
        for t in types:
            self.highlighting_rules.append((
                f'\\b{t}\\b', type_format
            ))

        # Level attributes
        level_format = QTextCharFormat()
        level_format.setForeground(QColor("#ffb86c"))  # Orange
        self.highlighting_rules.append((
            r'#\[level:\s*\d+\]', level_format
        ))

        # Functions
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#50fa7b"))  # Green
        self.highlighting_rules.append((
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*(?=\()', function_format
        ))

        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#bd93f9"))  # Purple
        self.highlighting_rules.append((
            r'\b\d+\.?\d*\b', number_format
        ))

        # Strings
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#f1fa8c"))  # Yellow
        self.highlighting_rules.append((
            r'"[^"\\]*(\\.[^"\\]*)*"', string_format
        ))
        self.highlighting_rules.append((
            r"'[^'\\]*(\\.[^'\\]*)*'", string_format
        ))

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6272a4"))  # Gray
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((
            r'#.*$', comment_format
        ))

        # Operators
        operator_format = QTextCharFormat()
        operator_format.setForeground(QColor("#ff79c6"))  # Pink
        self.highlighting_rules.append((
            r'[+\-*/=<>!&|^~%]+', operator_format
        ))

    def highlightBlock(self, text):
        import re
        for pattern, fmt in self.highlighting_rules:
            for match in re.finditer(pattern, text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, fmt)


# ═══════════════════════════════════════════════════════════════════════════════
#                              CODE EDITOR
# ═══════════════════════════════════════════════════════════════════════════════

class LineNumberArea(QWidget):
    """Line number area for the code editor"""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)


class CodeEditor(QPlainTextEdit):
    """Code editor with line numbers and syntax highlighting"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set font
        font = QFont("Consolas", 12)
        if not font.exactMatch():
            font = QFont("Courier New", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

        # Tab settings
        self.setTabStopDistance(QFontMetrics(font).horizontalAdvance(' ') * 4)

        # Line numbers
        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)
        self.update_line_number_area_width(0)

        # Syntax highlighter
        self.highlighter = TILHighlighter(self.document())

        # Current file
        self.file_path = None
        self.is_modified = False

        # Track modifications
        self.textChanged.connect(self.on_text_changed)

        # Style
        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #282a36;
                color: #f8f8f2;
                border: none;
                selection-background-color: #44475a;
            }
        """)

        self.highlight_current_line()

    def on_text_changed(self):
        self.is_modified = True

    def line_number_area_width(self):
        digits = len(str(max(1, self.blockCount())))
        return 10 + self.fontMetrics().horizontalAdvance('9') * digits

    def update_line_number_area_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(),
                                         self.line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            cr.left(), cr.top(),
            self.line_number_area_width(), cr.height()
        )

    def line_number_area_paint_event(self, event):
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor("#21222c"))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(
            self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QColor("#6272a4"))
                painter.drawText(
                    0, top,
                    self.line_number_area.width() - 5,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight, number
                )
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_number += 1

    def highlight_current_line(self):
        extra_selections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            line_color = QColor("#44475a")
            selection.format.setBackground(line_color)
            selection.format.setProperty(
                QTextCharFormat.Property.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extra_selections.append(selection)
        self.setExtraSelections(extra_selections)

    def keyPressEvent(self, event):
        # Auto-indent
        if event.key() == Qt.Key.Key_Return:
            cursor = self.textCursor()
            line = cursor.block().text()
            indent = len(line) - len(line.lstrip())

            # Increase indent after : or block starters
            if line.rstrip().endswith(':') or line.rstrip().endswith('()'):
                indent += 4

            super().keyPressEvent(event)
            self.insertPlainText(' ' * indent)
        else:
            super().keyPressEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
#                              OUTPUT CONSOLE
# ═══════════════════════════════════════════════════════════════════════════════

class OutputConsole(QPlainTextEdit):
    """Output console for displaying compilation results"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

        font = QFont("Consolas", 11)
        if not font.exactMatch():
            font = QFont("Courier New", 11)
        self.setFont(font)

        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1f29;
                color: #f8f8f2;
                border: none;
            }
        """)

    def write_output(self, text, color="#f8f8f2"):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(text, fmt)

        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def write_error(self, text):
        self.write_output(text, "#ff5555")

    def write_success(self, text):
        self.write_output(text, "#50fa7b")

    def write_info(self, text):
        self.write_output(text, "#8be9fd")

    def clear_output(self):
        self.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class TILMainWindow(QMainWindow):
    """Main IDE window"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("TIL IDE - Alisher Beisembekov")
        self.setGeometry(100, 100, 1400, 900)

        # Apply dark theme
        self.apply_dark_theme()

        # Create console first (needed by toolbar)
        self.console = OutputConsole()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create menu bar
        self.create_menu_bar()

        # Create toolbar
        self.create_toolbar()

        # Main splitter (horizontal)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # File explorer
        self.create_file_explorer()
        main_splitter.addWidget(self.file_explorer_frame)

        # Editor area (vertical splitter)
        editor_splitter = QSplitter(Qt.Orientation.Vertical)

        # Tab widget for multiple files
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: #282a36;
            }
            QTabBar::tab {
                background-color: #21222c;
                color: #f8f8f2;
                padding: 8px 16px;
                border: none;
            }
            QTabBar::tab:selected {
                background-color: #282a36;
                border-bottom: 2px solid #bd93f9;
            }
            QTabBar::tab:hover {
                background-color: #343746;
            }
        """)
        editor_splitter.addWidget(self.tab_widget)

        # Output console frame
        console_frame = QFrame()
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(0)

        console_header = QLabel("  OUTPUT")
        console_header.setStyleSheet("""
            QLabel {
                background-color: #21222c;
                color: #6272a4;
                padding: 5px;
                font-weight: bold;
            }
        """)
        console_layout.addWidget(console_header)
        console_layout.addWidget(self.console)
        editor_splitter.addWidget(console_frame)

        editor_splitter.setSizes([600, 200])
        main_splitter.addWidget(editor_splitter)

        main_splitter.setSizes([250, 1150])
        main_layout.addWidget(main_splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #191a21;
                color: #6272a4;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Process for running TIL
        self.process = None

        # Create welcome tab
        self.create_welcome_tab()

    def apply_dark_theme(self):
        """Apply Dracula-inspired dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #282a36;
            }
            QMenuBar {
                background-color: #21222c;
                color: #f8f8f2;
                border: none;
            }
            QMenuBar::item:selected {
                background-color: #44475a;
            }
            QMenu {
                background-color: #21222c;
                color: #f8f8f2;
                border: 1px solid #44475a;
            }
            QMenu::item:selected {
                background-color: #44475a;
            }
            QToolBar {
                background-color: #21222c;
                border: none;
                spacing: 5px;
                padding: 5px;
            }
            QToolButton {
                background-color: transparent;
                color: #f8f8f2;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QToolButton:hover {
                background-color: #44475a;
            }
            QToolButton:pressed {
                background-color: #6272a4;
            }
            QSplitter::handle {
                background-color: #191a21;
            }
            QScrollBar:vertical {
                background-color: #21222c;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #44475a;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        open_folder_action = QAction("Open &Folder...", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_file_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(lambda: self.get_current_editor().undo() if self.get_current_editor() else None)
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(lambda: self.get_current_editor().redo() if self.get_current_editor() else None)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        cut_action = QAction("Cu&t", self)
        cut_action.setShortcut(QKeySequence.StandardKey.Cut)
        cut_action.triggered.connect(lambda: self.get_current_editor().cut() if self.get_current_editor() else None)
        edit_menu.addAction(cut_action)

        copy_action = QAction("&Copy", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(lambda: self.get_current_editor().copy() if self.get_current_editor() else None)
        edit_menu.addAction(copy_action)

        paste_action = QAction("&Paste", self)
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        paste_action.triggered.connect(lambda: self.get_current_editor().paste() if self.get_current_editor() else None)
        edit_menu.addAction(paste_action)

        # Run menu
        run_menu = menubar.addMenu("&Run")

        run_action = QAction("&Run", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self.run_code)
        run_menu.addAction(run_action)

        build_action = QAction("&Build", self)
        build_action.setShortcut("F6")
        build_action.triggered.connect(self.build_code)
        run_menu.addAction(build_action)

        check_action = QAction("&Check Syntax", self)
        check_action.setShortcut("F7")
        check_action.triggered.connect(self.check_code)
        run_menu.addAction(check_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        docs_action = QAction("&Documentation", self)
        docs_action.triggered.connect(lambda: os.system("start https://til-dev.vercel.app"))
        help_menu.addAction(docs_action)

    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # New file
        new_btn = QPushButton("📄 New")
        new_btn.clicked.connect(self.new_file)
        new_btn.setStyleSheet(self.get_button_style())
        toolbar.addWidget(new_btn)

        # Open file
        open_btn = QPushButton("📂 Open")
        open_btn.clicked.connect(self.open_file)
        open_btn.setStyleSheet(self.get_button_style())
        toolbar.addWidget(open_btn)

        # Save file
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_file)
        save_btn.setStyleSheet(self.get_button_style())
        toolbar.addWidget(save_btn)

        toolbar.addSeparator()

        # Run
        run_btn = QPushButton("▶️ Run (F5)")
        run_btn.clicked.connect(self.run_code)
        run_btn.setStyleSheet(self.get_button_style("#50fa7b"))
        toolbar.addWidget(run_btn)

        # Build
        build_btn = QPushButton("🔨 Build (F6)")
        build_btn.clicked.connect(self.build_code)
        build_btn.setStyleSheet(self.get_button_style("#8be9fd"))
        toolbar.addWidget(build_btn)

        # Check
        check_btn = QPushButton("✓ Check (F7)")
        check_btn.clicked.connect(self.check_code)
        check_btn.setStyleSheet(self.get_button_style("#ffb86c"))
        toolbar.addWidget(check_btn)

        toolbar.addSeparator()

        # Clear console
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self.console.clear_output)
        clear_btn.setStyleSheet(self.get_button_style())
        toolbar.addWidget(clear_btn)

    def get_button_style(self, color=None):
        if color:
            return f"""
                QPushButton {{
                    background-color: {color};
                    color: #282a36;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {color}cc;
                }}
                QPushButton:pressed {{
                    background-color: {color}99;
                }}
            """
        return """
            QPushButton {
                background-color: #44475a;
                color: #f8f8f2;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #6272a4;
            }
            QPushButton:pressed {
                background-color: #44475a;
            }
        """

    def create_file_explorer(self):
        """Create the file explorer panel"""
        self.file_explorer_frame = QFrame()
        self.file_explorer_frame.setStyleSheet("""
            QFrame {
                background-color: #21222c;
                border: none;
            }
        """)

        layout = QVBoxLayout(self.file_explorer_frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("  EXPLORER")
        header.setStyleSheet("""
            QLabel {
                color: #6272a4;
                padding: 10px 5px;
                font-weight: bold;
            }
        """)
        layout.addWidget(header)

        # Current folder label
        self.folder_label = QLabel("  No folder open")
        self.folder_label.setStyleSheet("color: #f8f8f2; padding: 5px;")
        layout.addWidget(self.folder_label)

        # File list
        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self.on_file_item_clicked)
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: #21222c;
                color: #f8f8f2;
                border: none;
            }
            QListWidget::item:hover {
                background-color: #44475a;
            }
            QListWidget::item:selected {
                background-color: #44475a;
            }
        """)
        layout.addWidget(self.file_list)

        self.current_folder = None

    def refresh_file_list(self, folder_path):
        """Refresh the file list for the given folder"""
        self.file_list.clear()
        self.current_folder = folder_path
        self.folder_label.setText(f"  📁 {os.path.basename(folder_path)}")

        try:
            # Add parent directory option
            if os.path.dirname(folder_path) != folder_path:
                item = QListWidgetItem("📁 ..")
                item.setData(Qt.ItemDataRole.UserRole, os.path.dirname(folder_path))
                item.setData(Qt.ItemDataRole.UserRole + 1, "folder")
                self.file_list.addItem(item)

            # List directories first
            items = sorted(os.listdir(folder_path))
            dirs = [i for i in items if os.path.isdir(os.path.join(folder_path, i)) and not i.startswith('.')]
            files = [i for i in items if os.path.isfile(os.path.join(folder_path, i))]

            for d in dirs:
                item = QListWidgetItem(f"📁 {d}")
                item.setData(Qt.ItemDataRole.UserRole, os.path.join(folder_path, d))
                item.setData(Qt.ItemDataRole.UserRole + 1, "folder")
                self.file_list.addItem(item)

            for f in files:
                # Filter by TIL-related files
                if f.endswith(('.til', '.py', '.md', '.txt', '.json')):
                    icon = "📄" if f.endswith('.til') else "📝"
                    item = QListWidgetItem(f"{icon} {f}")
                    item.setData(Qt.ItemDataRole.UserRole, os.path.join(folder_path, f))
                    item.setData(Qt.ItemDataRole.UserRole + 1, "file")
                    self.file_list.addItem(item)
        except Exception as e:
            self.console.write_error(f"Error reading folder: {e}\n")

    def on_file_item_clicked(self, item):
        """Handle double-click on file list item"""
        path = item.data(Qt.ItemDataRole.UserRole)
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)

        if item_type == "folder":
            self.refresh_file_list(path)
        else:
            self.open_file_in_editor(path)

    def create_welcome_tab(self):
        """Create welcome tab"""
        welcome = QPlainTextEdit()
        welcome.setReadOnly(True)
        welcome.setPlainText("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║                        TIL IDE v1.0.0                            ║
║                                                                   ║
║              "Simpler than Python. Faster than C.                ║
║                      Smarter than all."                          ║
║                                                                   ║
║                   Author: Alisher Beisembekov                    ║
║                                                                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  QUICK START:                                                    ║
║                                                                   ║
║  • Press Ctrl+N to create a new file                             ║
║  • Press Ctrl+O to open an existing file                         ║
║  • Press F5 to run your code                                     ║
║  • Press F6 to build an executable                               ║
║  • Press F7 to check syntax                                      ║
║                                                                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  EXAMPLE CODE:                                                   ║
║                                                                   ║
║  main()                                                          ║
║      print("Hello, World!")                                      ║
║      let x = 42                                                  ║
║      print(x)                                                    ║
║                                                                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Website: https://til-dev.vercel.app                             ║
║  GitHub:  https://github.com/damn-glitch/TIL                     ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")
        welcome.setStyleSheet("""
            QPlainTextEdit {
                background-color: #282a36;
                color: #50fa7b;
                border: none;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 13px;
            }
        """)
        self.tab_widget.addTab(welcome, "Welcome")

    def open_file_in_editor(self, file_path):
        """Open a file in a new or existing tab"""
        # Check if already open
        for i in range(self.tab_widget.count()):
            editor = self.tab_widget.widget(i)
            if isinstance(editor, CodeEditor) and editor.file_path == file_path:
                self.tab_widget.setCurrentIndex(i)
                return

        # Create new tab
        editor = CodeEditor()
        editor.file_path = file_path

        with open(file_path, 'r', encoding='utf-8') as f:
            editor.setPlainText(f.read())

        editor.is_modified = False

        file_name = os.path.basename(file_path)
        self.tab_widget.addTab(editor, file_name)
        self.tab_widget.setCurrentWidget(editor)

        self.status_bar.showMessage(f"Opened: {file_path}")

    def get_current_editor(self) -> CodeEditor:
        """Get the current active editor"""
        widget = self.tab_widget.currentWidget()
        if isinstance(widget, CodeEditor):
            return widget
        return None

    def new_file(self):
        """Create a new file"""
        editor = CodeEditor()
        editor.setPlainText('# New TIL file\n\nmain()\n    print("Hello, World!")\n')
        self.tab_widget.addTab(editor, "untitled.til")
        self.tab_widget.setCurrentWidget(editor)

    def open_file(self):
        """Open file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open TIL File", "",
            "TIL Files (*.til);;All Files (*)"
        )
        if file_path:
            self.open_file_in_editor(file_path)

    def open_folder(self):
        """Open folder in explorer"""
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self.refresh_file_list(folder)
            self.status_bar.showMessage(f"Opened folder: {folder}")

    def save_file(self):
        """Save current file"""
        editor = self.get_current_editor()
        if not editor:
            return

        if editor.file_path:
            with open(editor.file_path, 'w', encoding='utf-8') as f:
                f.write(editor.toPlainText())
            editor.is_modified = False
            self.status_bar.showMessage(f"Saved: {editor.file_path}")
        else:
            self.save_file_as()

    def save_file_as(self):
        """Save file with new name"""
        editor = self.get_current_editor()
        if not editor:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save TIL File", "",
            "TIL Files (*.til);;All Files (*)"
        )
        if file_path:
            editor.file_path = file_path
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(editor.toPlainText())
            editor.is_modified = False

            # Update tab name
            idx = self.tab_widget.currentIndex()
            self.tab_widget.setTabText(idx, os.path.basename(file_path))
            self.status_bar.showMessage(f"Saved: {file_path}")

    def close_tab(self, index):
        """Close a tab"""
        editor = self.tab_widget.widget(index)
        if isinstance(editor, CodeEditor) and editor.is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Save changes before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Save:
                self.tab_widget.setCurrentIndex(index)
                self.save_file()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        self.tab_widget.removeTab(index)

    def run_code(self):
        """Run current TIL file"""
        editor = self.get_current_editor()
        if not editor:
            self.console.write_error("No file open\n")
            return

        # Save first
        self.save_file()

        if not editor.file_path:
            self.console.write_error("Please save the file first\n")
            return

        self.console.clear_output()
        self.console.write_info(f"▶ Running: {editor.file_path}\n")
        self.console.write_info("─" * 50 + "\n")

        # Run with TIL compiler
        try:
            result = subprocess.run(
                ["til", "run", editor.file_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout:
                self.console.write_output(result.stdout)
            if result.stderr:
                self.console.write_error(result.stderr)

            self.console.write_info("\n" + "─" * 50 + "\n")
            if result.returncode == 0:
                self.console.write_success("✓ Execution completed successfully\n")
            else:
                self.console.write_error(f"✗ Execution failed (exit code: {result.returncode})\n")

        except FileNotFoundError:
            self.console.write_error("TIL compiler not found. Please install TIL first.\n")
            self.console.write_info("Visit: https://til-dev.vercel.app\n")
        except subprocess.TimeoutExpired:
            self.console.write_error("Execution timed out (30s limit)\n")
        except Exception as e:
            self.console.write_error(f"Error: {e}\n")

        self.status_bar.showMessage("Run completed")

    def build_code(self):
        """Build current TIL file to executable"""
        editor = self.get_current_editor()
        if not editor:
            self.console.write_error("No file open\n")
            return

        self.save_file()

        if not editor.file_path:
            self.console.write_error("Please save the file first\n")
            return

        self.console.clear_output()
        self.console.write_info(f"🔨 Building: {editor.file_path}\n")
        self.console.write_info("─" * 50 + "\n")

        try:
            result = subprocess.run(
                ["til", "build", editor.file_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.stdout:
                self.console.write_output(result.stdout)
            if result.stderr:
                self.console.write_error(result.stderr)

            self.console.write_info("\n" + "─" * 50 + "\n")
            if result.returncode == 0:
                self.console.write_success("✓ Build completed successfully\n")
            else:
                self.console.write_error(f"✗ Build failed (exit code: {result.returncode})\n")

        except FileNotFoundError:
            self.console.write_error("TIL compiler not found.\n")
        except subprocess.TimeoutExpired:
            self.console.write_error("Build timed out (60s limit)\n")
        except Exception as e:
            self.console.write_error(f"Error: {e}\n")

        self.status_bar.showMessage("Build completed")

    def check_code(self):
        """Check syntax of current file"""
        editor = self.get_current_editor()
        if not editor:
            self.console.write_error("No file open\n")
            return

        self.save_file()

        if not editor.file_path:
            self.console.write_error("Please save the file first\n")
            return

        self.console.clear_output()
        self.console.write_info(f"✓ Checking: {editor.file_path}\n")
        self.console.write_info("─" * 50 + "\n")

        try:
            result = subprocess.run(
                ["til", "check", editor.file_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.stdout:
                self.console.write_output(result.stdout)
            if result.stderr:
                self.console.write_error(result.stderr)

            self.console.write_info("\n" + "─" * 50 + "\n")
            if result.returncode == 0:
                self.console.write_success("✓ No syntax errors found\n")
            else:
                self.console.write_error("✗ Syntax errors detected\n")

        except FileNotFoundError:
            self.console.write_error("TIL compiler not found.\n")
        except Exception as e:
            self.console.write_error(f"Error: {e}\n")

        self.status_bar.showMessage("Check completed")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About TIL IDE",
            """<h2>TIL IDE v1.0.0</h2>
            <p><b>Author:</b> Alisher Beisembekov</p>
            <p><b>Tagline:</b> "Simpler than Python. Faster than C. Smarter than all."</p>
            <hr>
            <p>TIL is a multi-level programming language designed for systems programming with safety guarantees.</p>
            <p><b>Website:</b> <a href="https://til-dev.vercel.app">til-dev.vercel.app</a></p>
            <p><b>GitHub:</b> <a href="https://github.com/damn-glitch/TIL">github.com/damn-glitch/TIL</a></p>
            """
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("TIL IDE")
    app.setOrganizationName("Alisher Beisembekov")

    # Set application-wide font
    font = app.font()
    font.setFamily("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)

    window = TILMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()