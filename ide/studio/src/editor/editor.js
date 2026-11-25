// Editor states and views: the CodeMirror extension stack of TIL Studio.
import { EditorState, Compartment } from '@codemirror/state';
import {
    EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection, dropCursor,
    rectangularSelection, crosshairCursor, highlightSpecialChars, scrollPastEnd,
} from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab, addCursorAbove, addCursorBelow } from '@codemirror/commands';
import { foldGutter, foldKeymap, indentOnInput, bracketMatching, syntaxHighlighting } from '@codemirror/language';
import { autocompletion, completionKeymap, closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete';
import { searchKeymap, highlightSelectionMatches, search } from '@codemirror/search';
import { lintKeymap } from '@codemirror/lint';
import { tilLanguage, tilHighlighter } from './til_lang.js';
import { studioTheme } from './theme.js';
import { studioGutters } from './gutters.js';
import { docContext, lspExtensions, completionSource, errorLensEnabled, lspFeatureToggles } from '../lsp/cm_lsp.js';
import { icon } from '../core/icons.js';

export const readOnlyComp = new Compartment();
export const optionsComp = new Compartment();
export const wrapComp = new Compartment();

const til = tilLanguage();

function foldMarker(open) {
    const el = document.createElement('span');
    el.className = 'cm-fold-marker';
    el.appendChild(icon(open ? 'chevronDown' : 'chevronRight', { size: 14 }));
    return el;
}

const shared = [
    highlightSpecialChars(),
    history(),
    drawSelection(),
    dropCursor(),
    EditorState.allowMultipleSelections.of(true),
    indentOnInput(),
    syntaxHighlighting(tilHighlighter),
    bracketMatching(),
    closeBrackets(),
    autocompletion({ override: [completionSource], activateOnTyping: true, maxRenderedOptions: 100, icons: true, closeOnBlur: true, interactionDelay: 40 }),
    rectangularSelection(),
    crosshairCursor(),
    highlightActiveLine(),
    highlightActiveLineGutter(),
    highlightSelectionMatches({ minSelectionLength: 2, maxMatches: 400 }),
    search({ top: true }),
    scrollPastEnd(),
    studioTheme,
    keymap.of([
        { key: 'Mod-Alt-ArrowUp', run: addCursorAbove },
        { key: 'Mod-Alt-ArrowDown', run: addCursorBelow },
        ...closeBracketsKeymap,
        ...defaultKeymap,
        ...searchKeymap,
        ...historyKeymap,
        ...foldKeymap,
        ...completionKeymap,
        ...lintKeymap,
        indentWithTab,
    ]),
];

const gutterStack = [studioGutters(), lineNumbers(), foldGutter({ markerDOM: foldMarker })];

/* options that follow the settings: error lens, inlay hints, code lenses */
export function optionExtensions(settings) {
    return [
        errorLensEnabled.of(Boolean(settings.errorLens)),
        lspFeatureToggles.of({ inlayHints: Boolean(settings.inlayHints), codeLens: Boolean(settings.codeLens) }),
    ];
}

/**
 * A new editor state for a document.
 *   ctx: {uri, path, client, app}  (docContext facet: LSP features use it)
 *   language: 'til' | 'plain'
 */
export function createEditorState({ text, ctx, readOnly = false, language = 'til', settings, listeners = [] }) {
    return EditorState.create({
        doc: text,
        extensions: [
            docContext.of(ctx),
            language === 'til' ? til : [],
            shared,
            gutterStack,
            language === 'til' ? lspExtensions() : [],
            readOnlyComp.of([EditorState.readOnly.of(readOnly), EditorView.editable.of(!readOnly || language === 'til')]),
            optionsComp.of(optionExtensions(settings || {})),
            wrapComp.of(settings && settings.wordWrap ? EditorView.lineWrapping : []),
            listeners,
        ],
    });
}

export function createView(parent, state) {
    return new EditorView({ state, parent });
}

/* reconfigure the setting-driven parts of a state */
export function settingsEffects(settings) {
    return [
        optionsComp.reconfigure(optionExtensions(settings)),
        wrapComp.reconfigure(settings.wordWrap ? EditorView.lineWrapping : []),
    ];
}
