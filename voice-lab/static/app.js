const $ = id => document.getElementById(id);
let voices = [], history = [], busy = false, previewUrl;
const SETTINGS_KEY = 'voiceLab.chatterbox.settings.v1';
const SETTINGS_VERSION = 1;

const CHATTERBOX_DEFAULTS = {
  exaggeration: 0.5,
  cfg_weight: 0.5,
  temperature: 0.8,
  repetition_penalty: 1.2,
  top_p: 1.0,
  min_p: 0.05,
  max_new_tokens: 1500,
  lang_code: 'ru',
  seed: null,
  verbose: false
};

const CHATTERBOX_BOUNDS = {
  exaggeration: { min: 0, max: 1 },
  cfg_weight: { min: 0, max: 2 },
  temperature: { min: 0.1, max: 1.5 },
  repetition_penalty: { min: 1, max: 2 },
  top_p: { min: 0.01, max: 1 },
  min_p: { min: 0, max: 1 },
  max_new_tokens: { min: 100, max: 3000 },
  seed: { min: 0, max: 4294967295 }
};

function engineOptions() {
  const engine = $('engine').value;
  if (engine === 'xtts' && !$('acceptXtts').checked) throw new Error('Для XTTS сначала прочитайте лицензию и подтвердите её условия.');
  const opts = { engine, exaggeration: Number($('exaggeration').value), xtts_license_accepted: $('acceptXtts').checked };
  if (engine === 'chatterbox') {
    const settings = collectChatterboxSettings();
    const validation = validateChatterboxSettings(settings);
    if (!validation.valid) throw new Error(validation.error);
    opts.chatterbox = settings;
  }
  return opts;
}

function collectChatterboxSettings() {
  return {
    exaggeration: Number($('exaggeration').value),
    cfg_weight: Number($('cfg_weight').value),
    temperature: Number($('temperature').value),
    repetition_penalty: Number($('repetition_penalty').value),
    top_p: Number($('top_p').value),
    min_p: Number($('min_p').value),
    max_new_tokens: Number($('max_new_tokens').value),
    lang_code: $('lang_code').value,
    seed: $('seed').value === '' ? null : Number($('seed').value),
    verbose: $('verbose').checked
  };
}

function validateChatterboxSettings(settings) {
  for (const [key, bounds] of Object.entries(CHATTERBOX_BOUNDS)) {
    const value = settings[key];
    if (key === 'seed') {
      if (value !== null && (typeof value !== 'number' || !Number.isInteger(value) || value < bounds.min || value > bounds.max)) {
        return { valid: false, error: `Seed должно быть целым числом от ${bounds.min} до ${bounds.max} или пустым.` };
      }
      continue;
    }
    if (key === 'max_new_tokens') {
      if (typeof value !== 'number' || !Number.isInteger(value) || value < bounds.min || value > bounds.max) {
        return { valid: false, error: `${key} должно быть целым числом от ${bounds.min} до ${bounds.max}.` };
      }
      continue;
    }
    if (typeof value !== 'number' || Number.isNaN(value) || !Number.isFinite(value) || value < bounds.min || value > bounds.max) {
      return { valid: false, error: `${key} должно быть числом от ${bounds.min} до ${bounds.max}.` };
    }
  }
  return { valid: true };
}

function applyChatterboxSettings(settings) {
  for (const [key, value] of Object.entries(settings)) {
    const el = $(key);
    if (!el) continue;
    if (el.type === 'checkbox') {
      el.checked = Boolean(value);
    } else if (el.tagName === 'SELECT') {
      el.value = value;
    } else if (el.type === 'range') {
      el.value = value;
      const output = $(key.replace('_', '') + 'Value') || $(key + 'Value');
      if (output) output.textContent = value;
    } else {
      el.value = value === null ? '' : value;
    }
  }
  $('expressionValue').textContent = $('exaggeration').value;
}

function saveChatterboxSettings() {
  try {
    const settings = collectChatterboxSettings();
    const validation = validateChatterboxSettings(settings);
    if (!validation.valid) return false;
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({ v: SETTINGS_VERSION, data: settings }));
    return true;
  } catch {
    return false;
  }
}

function loadChatterboxSettings() {
  try {
    const stored = localStorage.getItem(SETTINGS_KEY);
    if (!stored) return false;
    const parsed = JSON.parse(stored);
    if (parsed.v !== SETTINGS_VERSION) return false;
    const settings = parsed.data;
    const validation = validateChatterboxSettings(settings);
    if (!validation.valid) return false;
    applyChatterboxSettings(settings);
    return true;
  } catch {
    return false;
  }
}

function resetChatterboxSettings() {
  applyChatterboxSettings(CHATTERBOX_DEFAULTS);
  saveChatterboxSettings();
}

function showEngine() {
  const id = $('engine').value;
  $('expressionSettings').hidden = id !== 'chatterbox';
  $('chatterboxAdvanced').hidden = id !== 'chatterbox';
  $('xttsLicense').hidden = id !== 'xtts';
  $('engineHint').textContent = {
    chatterbox: 'Русский язык · клонирование + настройка выразительности. Расшифровка не используется.',
    qwen: 'Русский язык · расшифровка помогает передать манеру речи. Интонация задаётся образцом.',
    xtts: 'Русский язык · клонирование по записи. Расшифровка не используется. Загрузка модели при каждой генерации может занять время.',
  }[id];
  updateVoiceInfo();
  refreshStatus();
}
$('engine').addEventListener('change', showEngine);

$('exaggeration').addEventListener('input', () => {
  $('expressionValue').textContent = $('exaggeration').value;
  saveChatterboxSettings();
});
['cfg_weight', 'temperature', 'repetition_penalty', 'top_p', 'min_p', 'max_new_tokens', 'lang_code', 'seed', 'verbose'].forEach(key => {
  const el = $(key);
  if (el) {
    el.addEventListener('input', () => {
      const output = $(key.replace('_', '') + 'Value') || $(key + 'Value');
      if (output) output.textContent = el.value;
      saveChatterboxSettings();
    });
    if (el.type === 'number') {
      el.addEventListener('change', () => saveChatterboxSettings());
    }
  }
});

$('resetChatterbox').addEventListener('click', () => {
  resetChatterboxSettings();
  setNotice('Настройки Chatterbox сброшены к значениям по умолчанию.');
});

const setNotice = (text, error = false) => {
  $('notice').textContent = text;
  $('notice').className = error ? 'error' : '';
  $('notice').hidden = !text;
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Проверьте заполненные поля.');
  return result;
}
const post = (path, data) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });

async function refreshStatus() {
  try {
    const s = await api('/api/status');
    if (s.tts.xtts_license_accepted) $('acceptXtts').checked = true;
    $('ollamaStatus').textContent = s.ollama.error ? '○ Ollama недоступна' : `● Ollama · ${s.ollama.models.length} локальная модель`;
    const labels = { not_loaded: 'Озвучка ещё не загружена', loading: 'Скачивание / загрузка озвучки…', ready: 'Озвучка готова', error: 'Ошибка загрузки озвучки' };
    const selected = $('engine').value;
    const ready = s.tts.engine === selected && s.tts.state === 'ready';
    const label = s.tts.engine === selected ? (labels[s.tts.state] || s.tts.state) : 'Движок загрузится при запуске';
    $('ttsStatus').textContent = `${ready ? '●' : '○'} ${label}`;
    $('ttsStatus').title = s.tts.model;
    const old = $('model').value;
    $('model').replaceChildren(...s.ollama.models.map(name => new Option(name, name)));
    if (s.ollama.models.includes(old)) $('model').value = old;
    else if (s.ollama.models.includes('gemma4:e2b')) $('model').value = 'gemma4:e2b';
    if (!s.ollama.models.length) $('model').add(new Option('Нет локальных моделей', ''));
    $('prepare').textContent = ready ? 'Движок готов' : 'Подготовить озвучку';
    $('prepare').disabled = busy || s.busy || ready;
    if (!s.ffmpeg) setNotice('Для обработки записи нужен FFmpeg. Инструкция есть в README.', true);
  } catch { $('ollamaStatus').textContent = '○ Сервер недоступен'; }
}

function setBusy(value) {
  busy = value;
  const ids = ['speak', 'ask', 'upload', 'prepare', 'clear', 'voices', 'model', 'saveTranscript', 'savedTranscript', 'engine', 'exaggeration', 'acceptXtts',
    'cfg_weight', 'temperature', 'repetition_penalty', 'top_p', 'min_p', 'max_new_tokens', 'lang_code', 'seed', 'verbose', 'resetChatterbox'];
  for (const id of ids) if ($(id)) $(id).disabled = value;
  document.querySelectorAll('.retry').forEach(el => { el.disabled = value; });
}

async function action(callback) {
  if (busy) return;
  setBusy(true);
  try { await callback(); }
  catch (error) { setNotice(error.message || 'Не удалось выполнить запрос.', true); }
  finally { setBusy(false); await refreshStatus(); }
}

async function waitJob(job) {
  const started = Date.now();
  while (job.state === 'running') {
    await new Promise(resolve => setTimeout(resolve, 1500));
    job = await api(`/api/jobs/${job.id}`);
    const seconds = Math.round((Date.now() - started) / 1000);
    setNotice(`Озвучка работает · ${seconds} сек. Первый запуск включает загрузку модели в несколько гигабайт. Это может занять несколько минут.`);
  }
  if (job.state === 'error') throw new Error(job.error);
  return job;
}

function selectedVoice() {
  if (!$('voices').value) throw new Error('Сначала загрузите и сохраните запись голоса.');
  return $('voices').value;
}

function showVoice() {
  const voice = voices.find(v => v.id === $('voices').value);
  $('voicePreview').hidden = !voice;
  updateVoiceInfo();
  if (voice) {
    $('voicePreview').src = voice.audioUrl;
    $('savedTranscript').value = voice.transcript || '';
  }
}

function updateVoiceInfo() {
  const voice = voices.find(v => v.id === $('voices').value);
  const qwen = $('engine').value === 'qwen';
  $('savedVoiceEditor').hidden = !voice || !qwen;
  if (voice) $('voiceInfo').textContent = `${voice.duration} сек. · ${qwen ? (voice.transcript ? 'Голосовой пример + текст' : 'Только тембр: добавь расшифровку ниже') : 'Клонирование по аудио'} · хранится локально`;
}

async function refreshVoices(selected) {
  voices = await api('/api/voices');
  const old = selected || $('voices').value;
  $('voices').replaceChildren(...voices.map(v => new Option(v.name, v.id)));
  if (!voices.length) $('voices').add(new Option('Пока нет записей', ''));
  else if (voices.some(v => v.id === old)) $('voices').value = old;
  showVoice();
}
$('voices').addEventListener('change', showVoice);

$('saveTranscript').addEventListener('click', () => action(async () => {
  const voice = selectedVoice();
  const result = await api(`/api/voices/${voice}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ transcript: $('savedTranscript').value }) });
  await refreshVoices(voice);
  setNotice(result.transcript ? 'Расшифровка сохранена. Следующая озвучка использует голосовой пример вместе с текстом.' : 'Расшифровка очищена. Будет использоваться только образец тембра.');
}));

$('file').addEventListener('change', () => {
  const file = $('file').files[0];
  if (!file) return;
  $('fileName').textContent = file.name;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('sourcePreview').src = previewUrl;
  $('sourcePreview').hidden = false;
});
for (const event of ['dragenter', 'dragover']) $('dropzone').addEventListener(event, () => $('dropzone').classList.add('dragging'));
for (const event of ['dragleave', 'drop']) $('dropzone').addEventListener(event, () => $('dropzone').classList.remove('dragging'));

$('uploadForm').addEventListener('submit', event => {
  event.preventDefault();
  action(async () => {
    const file = $('file').files[0];
    if (!file) throw new Error('Выберите запись голоса.');
    if (file.size > 30 * 1024 * 1024) throw new Error('Максимальный размер записи — 30 МБ.');
    const data = new FormData();
    data.append('file', file);
    for (const [key, id] of [['name', 'voiceName'], ['start', 'start'], ['seconds', 'seconds'], ['transcript', 'transcript']]) data.append(key, $(id).value);
    setNotice('Подготавливаем выбранный фрагмент записи…');
    const voice = await api('/api/voices', { method: 'POST', body: data });
    await refreshVoices(voice.id);
    setNotice('Голос сохранён. Прослушайте фрагмент и нажмите «Озвучить».');
  });
});

$('prepare').addEventListener('click', () => action(async () => {
  setNotice('Подготавливаем голосовую модель. Первая загрузка требует интернета.');
  await waitJob(await post('/api/tts/prepare', engineOptions()));
  setNotice('Голосовая модель готова. Можно озвучивать текст.');
}));

function showTestParams(settings) {
  const params = $('testParams');
  params.closest('details').hidden = !settings;
  params.textContent = settings ? JSON.stringify(settings, null, 2) : '';
  if (settings) params.dataset.settings = JSON.stringify(settings);
  else delete params.dataset.settings;
}

$('speak').addEventListener('click', () => action(async () => {
  const voice = selectedVoice(), text = $('testText').value.trim();
  if (!text) throw new Error('Введите текст для озвучки.');
  setNotice('Создаём аудио твоим голосом…');
  const result = await waitJob(await post('/api/speech', { text, voice, ...engineOptions() }));
  $('testAudio').src = result.audioUrl;
  $('testDownload').href = result.audioUrl;
  $('testTiming').textContent = `${result.engineName} · Генерация: ${result.seconds} сек. · Аудио: ${result.duration} сек.`;
  showTestParams(result.settings);
  $('testResult').hidden = false;
  $('testEmpty').hidden = true;
  setNotice('Готово. Нажмите воспроизведение и сравните голос с образцом.');
}));

$('copyTestParams').addEventListener('click', () => {
  const params = $('testParams').dataset.settings;
  if (params) {
    navigator.clipboard.writeText(params);
    setNotice('Параметры скопированы в буфер обмена.');
  }
});

for (const mode of ['test', 'chat']) $(mode + 'Tab').addEventListener('click', () => {
  for (const other of ['test', 'chat']) {
    $(other + 'Tab').setAttribute('aria-selected', String(other === mode));
    $(other + 'Panel').hidden = other !== mode;
  }
});

function addMessage(role, text) {
  $('chatEmpty')?.remove();
  const box = document.createElement('div'); box.className = `message ${role}`;
  const label = document.createElement('small'); label.textContent = role === 'user' ? 'ТЫ' : 'GEMMA';
  const content = document.createElement('p'); content.textContent = text;
  box.append(label, content); $('messages').append(box); box.scrollIntoView({ block: 'nearest' });
  return box;
}

async function speakMessage(box, text, voice) {
  setNotice('Ответ готов. Создаём озвучку…');
  const result = await waitJob(await post('/api/speech', { text, voice, ...engineOptions() }));
  box.querySelector('audio')?.remove();
  const audio = document.createElement('audio'); audio.controls = true; audio.src = result.audioUrl;
  box.append(audio);
  let timing = box.querySelector('.speech-timing');
  if (!timing) { timing = document.createElement('span'); timing.className = 'hint speech-timing'; box.append(timing); }
  timing.textContent = `${result.engineName} · Озвучка: ${result.seconds} сек. · ${result.duration} сек. аудио`;
  if (result.settings) {
    const details = document.createElement('details');
    details.className = 'generation-params';
    details.innerHTML = `<summary>Параметры этой генерации</summary><pre class="params-json">${JSON.stringify(result.settings, null, 2)}</pre><button class="link-button copy-params" type="button">Копировать JSON</button>`;
    details.querySelector('.copy-params').addEventListener('click', () => {
      navigator.clipboard.writeText(JSON.stringify(result.settings, null, 2));
      setNotice('Параметры скопированы в буфер обмена.');
    });
    box.append(details);
  }
  setNotice('Ответ озвучен. Нажмите воспроизведение.');
}

$('chatForm').addEventListener('submit', event => {
  event.preventDefault();
  action(async () => {
    const message = $('question').value.trim(), model = $('model').value;
    if (!message) throw new Error('Напишите вопрос.');
    if (!model) throw new Error('Откройте Ollama и установите локальную модель.');
    const auto = $('autoSpeech').checked;
    if (auto) engineOptions();
    const voice = auto ? selectedVoice() : null;
    const userBox = addMessage('user', message);
    setNotice('Gemma думает… Первый ответ может занять больше времени.');
    let result;
    try { result = await post('/api/chat', { message, model, history: history.slice(-20) }); }
    catch (error) { userBox.remove(); throw error; }
    $('question').value = '';
    history.push({ role: 'user', content: message }, { role: 'assistant', content: result.text });
    history = history.slice(-20);
    const box = addMessage('assistant', result.text);
    const timing = document.createElement('span'); timing.className = 'hint'; timing.textContent = `Ответ: ${result.seconds} сек.`;
    const retry = document.createElement('button'); retry.className = 'link-button retry'; retry.textContent = 'Озвучить ещё раз ↗'; retry.disabled = true;
    retry.addEventListener('click', () => action(() => speakMessage(box, result.text, selectedVoice())));
    box.append(timing, retry);
    setNotice('Ответ готов.');
    if (auto) await speakMessage(box, result.text, voice);
  });
});

$('clear').addEventListener('click', () => { history = []; $('messages').replaceChildren(); setNotice('История чата очищена.'); });

async function initSettings() {
  try {
    const response = await api('/api/tts/settings');
    const select = $('lang_code');
    select.replaceChildren(...Object.entries(response.languages).map(([code, name]) => new Option(`${name} (${code})`, code)));
    if (!loadChatterboxSettings()) {
      applyChatterboxSettings(CHATTERBOX_DEFAULTS);
    }
  } catch {
    const select = $('lang_code');
    select.replaceChildren(...Object.entries({
      "ar": "Арабский", "da": "Датский", "de": "Немецкий", "el": "Греческий", "en": "Английский",
      "es": "Испанский", "fi": "Финский", "fr": "Французский", "he": "Иврит", "hi": "Хинди",
      "it": "Итальянский", "ja": "Японский", "ko": "Корейский", "ms": "Малайский", "nl": "Нидерландский",
      "no": "Норвежский", "pl": "Польский", "pt": "Португальский", "ru": "Русский", "sv": "Шведский",
      "sw": "Суахили", "tr": "Турецкий", "zh": "Китайский"
    }).map(([code, name]) => new Option(`${name} (${code})`, code)));
    loadChatterboxSettings();
  }
}

showEngine();
initSettings().then(() => refreshVoices()).catch(error => setNotice(error.message, true));
setInterval(() => { if (!busy) refreshStatus(); }, 10000);
