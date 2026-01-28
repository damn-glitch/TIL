const vscode = require('vscode');
const { exec } = require('child_process');
const path = require('path');

/**
 * TIL Language Extension
 * Author: Alisher Beisembekov
 */

let outputChannel;

function activate(context) {
    console.log('TIL Language extension activated');
    
    // Create output channel
    outputChannel = vscode.window.createOutputChannel('TIL');
    
    // Register Run command (F5)
    let runCommand = vscode.commands.registerCommand('til.run', () => {
        runTILFile('run');
    });
    
    // Register Build command (F6)
    let buildCommand = vscode.commands.registerCommand('til.build', () => {
        runTILFile('build');
    });
    
    // Register Check command (F7)
    let checkCommand = vscode.commands.registerCommand('til.check', () => {
        runTILFile('check');
    });
    
    // Register Run in Terminal command
    let runInTerminalCommand = vscode.commands.registerCommand('til.runInTerminal', () => {
        runTILInTerminal();
    });
    
    context.subscriptions.push(runCommand, buildCommand, checkCommand, runInTerminalCommand);
    context.subscriptions.push(outputChannel);
    
    // Show welcome message on first activation
    const config = vscode.workspace.getConfiguration('til');
    if (!config.get('welcomeShown')) {
        vscode.window.showInformationMessage(
            'TIL Language extension activated! Press F5 to run, F6 to build, F7 to check.',
            'Got it'
        );
    }
}

function runTILFile(command) {
    const editor = vscode.window.activeTextEditor;
    
    if (!editor) {
        vscode.window.showErrorMessage('No file open');
        return;
    }
    
    const document = editor.document;
    
    if (document.languageId !== 'til') {
        vscode.window.showErrorMessage('Not a TIL file');
        return;
    }
    
    // Save the file first
    document.save().then(() => {
        const filePath = document.fileName;
        const fileName = path.basename(filePath);
        const workDir = path.dirname(filePath);
        
        // Get compiler path from settings
        const config = vscode.workspace.getConfiguration('til');
        const compilerPath = config.get('compilerPath', 'til');
        const optimization = config.get('optimization', '-O2');
        
        // Build command
        let cmd;
        if (command === 'run') {
            cmd = `"${compilerPath}" run "${filePath}"`;
        } else if (command === 'build') {
            cmd = `"${compilerPath}" build "${filePath}" ${optimization}`;
        } else if (command === 'check') {
            cmd = `"${compilerPath}" check "${filePath}"`;
        }
        
        // Show output channel
        outputChannel.clear();
        outputChannel.show(true);
        outputChannel.appendLine(`▶ ${command.toUpperCase()}: ${fileName}`);
        outputChannel.appendLine('─'.repeat(50));
        
        // Execute command
        exec(cmd, { cwd: workDir, timeout: 30000 }, (error, stdout, stderr) => {
            if (stdout) {
                outputChannel.appendLine(stdout);
            }
            if (stderr) {
                outputChannel.appendLine(stderr);
            }
            
            outputChannel.appendLine('─'.repeat(50));
            
            if (error) {
                outputChannel.appendLine(`✗ ${command} failed (exit code: ${error.code})`);
                
                // Parse errors and show diagnostics
                if (stderr) {
                    showDiagnostics(document, stderr);
                }
            } else {
                outputChannel.appendLine(`✓ ${command} completed successfully`);
                clearDiagnostics(document);
            }
        });
    });
}

function runTILInTerminal() {
    const editor = vscode.window.activeTextEditor;
    
    if (!editor) {
        vscode.window.showErrorMessage('No file open');
        return;
    }
    
    const document = editor.document;
    
    if (document.languageId !== 'til') {
        vscode.window.showErrorMessage('Not a TIL file');
        return;
    }
    
    // Save and run in terminal
    document.save().then(() => {
        const filePath = document.fileName;
        const config = vscode.workspace.getConfiguration('til');
        const compilerPath = config.get('compilerPath', 'til');
        
        // Create or reuse terminal
        let terminal = vscode.window.terminals.find(t => t.name === 'TIL');
        if (!terminal) {
            terminal = vscode.window.createTerminal('TIL');
        }
        
        terminal.show();
        terminal.sendText(`${compilerPath} run "${filePath}"`);
    });
}

// Diagnostics collection for error highlighting
const diagnosticCollection = vscode.languages.createDiagnosticCollection('til');

function showDiagnostics(document, errorText) {
    const diagnostics = [];
    
    // Parse error messages like "Error at line 5: unexpected token"
    const errorRegex = /(?:error|Error).*?(?:line\s*)?(\d+).*?[:\s]+(.+)/gi;
    let match;
    
    while ((match = errorRegex.exec(errorText)) !== null) {
        const lineNum = parseInt(match[1]) - 1;
        const message = match[2].trim();
        
        if (lineNum >= 0 && lineNum < document.lineCount) {
            const line = document.lineAt(lineNum);
            const range = new vscode.Range(lineNum, 0, lineNum, line.text.length);
            const diagnostic = new vscode.Diagnostic(range, message, vscode.DiagnosticSeverity.Error);
            diagnostic.source = 'TIL';
            diagnostics.push(diagnostic);
        }
    }
    
    diagnosticCollection.set(document.uri, diagnostics);
}

function clearDiagnostics(document) {
    diagnosticCollection.delete(document.uri);
}

function deactivate() {
    diagnosticCollection.clear();
    diagnosticCollection.dispose();
}

module.exports = {
    activate,
    deactivate
};
