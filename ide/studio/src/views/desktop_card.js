// The web face: features of the desktop server show a friendly card.
import { h } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { customDialog } from '../ui/widgets.js';
import { desktopCard } from '../panels/debug.js';

export function desktopOnlyDialog(app, feature) {
    return customDialog({
        title: t('desktop.dialogTitle'),
        body: h('div', { class: 'dialog-body' }, desktopCard(app, feature)),
        buttons: [{ label: t('dialog.close'), value: true, primary: true }],
    });
}
