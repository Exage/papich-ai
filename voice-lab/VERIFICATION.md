# Verification Report

## Tests Executed

All 20 tests in `tests/test_api.py` pass:

```
.venv/bin/python -m pytest tests -q
....................
20 passed, 1 warning in 0.95s
```

### Test Coverage

| Test | Purpose |
| --- | --- |
| `test_upload_crop_and_retrieve` | Voice upload, crop, retrieval |
| `test_reject_invalid_and_short_audio_without_orphans` | Invalid/short audio rejection |
| `test_validation_and_origin_protection` | Input validation, CSRF protection |
| `test_chat_disallows_nonlocal_model` | Cloud model rejection |
| `test_failed_speech_releases_lock_and_reports_error` | Error handling, lock release |
| `test_engine_validation_and_xtts_consent` | Engine validation, XTTS consent |
| `test_engine_routing_and_metadata` | Engine routing (qwen, chatterbox, xtts) |
| `test_chatterbox_settings_endpoint` | `/api/tts/settings` endpoint |
| `test_speech_without_new_settings_uses_defaults` | Backward compatibility |
| `test_old_exaggeration_field_still_works` | Legacy `exaggeration` field |
| `test_chatterbox_exaggeration_priority_over_old_field` | Priority: chatterbox.exaggeration > exaggeration |
| `test_all_chatterbox_settings_reach_adapter` | All 10 parameters reach adapter |
| `test_job_metadata_contains_applied_settings` | Job metadata includes applied settings |
| `test_auto_seed_becomes_concrete_number` | Auto seed becomes concrete int |
| `test_invalid_chatterbox_values_rejected` | Boundary validation (18 cases) |
| `test_unknown_field_in_chatterbox_rejected` | Extra field rejection (forbid) |
| `test_chatterbox_settings_not_passed_to_qwen_or_xtts` | Settings isolation |
| `test_generation_error_releases_lock` | Lock release on error |

## Real Generation Check

- **Performed**: No (requires installed model and saved voice)
- **Audio evaluated by ear**: No
- **Environment**: Tests use mocked `engine.synthesize` to avoid model downloads

## Known Limitations

1. **Seed reproducibility**: Bit-exact reproducibility across devices/versions not guaranteed (MLX random state).
2. **Streaming**: Not supported by MLX Chatterbox implementation.
3. **Speed control**: Ignored by model.
4. **Emotion presets**: No discrete emotion modes — exaggeration only amplifies delivery.
5. **Voice parameter**: Ignored (uses reference audio).
6. **Language support**: Limited to 23 languages in `SUPPORTED_LANGUAGES`.
7. **Token limit**: `max_new_tokens` limits tokens, not seconds; low values may truncate.
8. **XTTS**: Separate process, reloads weights each request; requires license consent.
9. **Parallel generation**: Not supported (single worker, global lock).
10. **Model switching**: Unloads previous model; first load downloads weights.

## Files Changed

| File | Changes |
| --- | --- |
| `app.py` | Added `ChatterboxSettings` model, `CHATTERBOX_DEFAULTS`, `CHATTERBOX_BOUNDS`, `SUPPORTED_LANGUAGES`; new `/api/tts/settings` endpoint; updated `EngineRequest`, `SpeechRequest` with `chatterbox` field; priority logic for `exaggeration`; job metadata includes applied settings; unknown field rejection |
| `voice_engine.py` | Added `CHATTERBOX_GENERATE_ARGS`; updated `synthesize()` signature to accept `chatterbox_settings`; seed handling via `mlx.random.seed()`; returns `(duration, used_seed)`; filters allowed args for `generate()` |
| `static/index.html` | Added "Расширенные настройки Chatterbox" collapsible section with 10 fields; language dropdown; seed input; verbose checkbox; reset button; generation params display with copy button |
| `static/app.js` | Settings collector/validator; localStorage persistence (`voiceLab.chatterbox.settings.v1`); versioned migration; auto-load on startup; engine-aware show/hide; keyboard accessible; disabled during generation |
| `static/style.css` | Added `.settings-grid`, `.setting-row`, `.generation-params` styles; responsive single-column on narrow screens |
| `tests/test_api.py` | Added 10 new tests for Chatterbox settings validation, priority, metadata, seed, rejection, isolation |
| `README.md` | Documented all 10 parameters with defaults, bounds, descriptions; noted MLX limitations; updated API docs |

## Run Command

```bash
cd /Users/n.gorkavchuk/Desktop/papich-ai/voice-lab
bash run.command
# or
.venv/bin/python app.py
```

Then open http://127.0.0.1:8765

## Server Restart Required

Yes — Python files changed. Restart the server to apply changes.