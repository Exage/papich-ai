const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function ui() {
  const root = path.join(__dirname, '..', 'static');
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
  new vm.Script(source); // Check the entire file, including startup code.
  const details = { hidden: false };
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], {
    value: '', dataset: {}, addEventListener() {}, closest: () => details,
  }]));
  const context = vm.createContext({document: { getElementById: id => elements.get(id) || null }});
  // Register real handlers without starting polling or network requests.
  vm.runInContext(source.split('\nshowEngine();')[0], context);
  const defaults = vm.runInContext('CHATTERBOX_DEFAULTS', context);
  for (const [key, value] of Object.entries(defaults)) elements.get(key).value = value ?? '';
  return { context, elements, details };
}

test('integer inputs are not silently truncated', () => {
  const { context, elements } = ui();
  const validate = () => vm.runInContext('validateChatterboxSettings(collectChatterboxSettings()).valid', context);
  elements.get('max_new_tokens').value = '1500.9';
  assert.equal(validate(), false);
  elements.get('max_new_tokens').value = '1500';
  elements.get('seed').value = '42.9';
  assert.equal(validate(), false);
  elements.get('seed').value = '1e3';
  assert.equal(validate(), true);
  assert.equal(vm.runInContext('collectChatterboxSettings().seed', context), 1000);
  elements.get('seed').value = '';
  assert.equal(vm.runInContext('collectChatterboxSettings().seed', context), null);
});

test('results without settings clear previous metadata and hide the panel', () => {
  const { context, elements, details } = ui();
  vm.runInContext('showTestParams({seed: 42})', context);
  assert.equal(details.hidden, false);
  assert.equal(elements.get('testParams').dataset.settings, '{"seed":42}');
  vm.runInContext('showTestParams(undefined)', context);
  assert.equal(details.hidden, true);
  assert.equal(elements.get('testParams').textContent, '');
  assert.equal(elements.get('testParams').dataset.settings, undefined);
  vm.runInContext('showTestParams({seed: 7})', context);
  assert.equal(details.hidden, false);
  assert.equal(elements.get('testParams').dataset.settings, '{"seed":7}');
});
