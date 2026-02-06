# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- MJProxy provider support for local Midjourney via MJAPI:
  - Image tools: Midjourney V7, Midjourney V6, Niji
  - Video tool: Midjourney Video
- Automatic post-processing for MJProxy tasks:
  - Images: after `IMAGINE`, auto-run `U1` to return a single upscaled image (instead of the 2x2 grid)
  - Videos: if the initial `VIDEO` task returns only a preview, auto-run `U1` (e.g. `video_virtual_upscale`) to obtain the final mp4

### Changed

- Provider enablement logic:
  - MJProxy tools can be listed/registered when `mjproxy.url` is configured (even if `api_key` is empty)
- MJProxy settings UX:
  - `api_key` field is available (used as the HTTP `Authorization` header when the MJProxy server requires auth)
- MJProxy video submit defaults:
  - Default `batchSize=1` (bs1)
  - Explicitly submit `action=VIDEO` to avoid server-side defaults selecting extension modes

### Fixed

- Config sanitation:
  - Strip whitespace from provider `url` and `api_key` on load/save to prevent invalid HTTP headers (e.g. `Illegal header value`)
  - Strip text model `api_key` before instantiating ChatOpenAI
- MJProxy video URL handling:
  - More robust URL extraction with fallbacks (`videoUrl`, `videoUrls`, `url`, `imageUrl`, `properties`)
  - Polling continues until the URL is actually available

### Docs

- Clarify source run/start instructions (dev, production-like, browser-only):
  - `README.md`
  - `README_zh.md`
  - `README-zh.md`
