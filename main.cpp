// muzak visualizer: a music-driven flight through a small constellation.
//
// A camera soars through a small constellation of star systems, linked by a
// tree of hyperlanes and laid out differently for every song. Each star
// listens to one band of the spectrum, and its planets and moons whirl faster
// as that band gets louder. Squadrons of ships cruise from star to star
// trailing ribbons of light, stirring a stable-fluids simulation whose dye is
// drawn as the nebula. Every low-end hit sends a ring out from the centre star.
//
// Deps: SDL2 (window + audio), OpenGL 3.3 core, ffmpeg CLI (decode + mp4 export).

#define GL_SILENCE_DEPRECATION
#include <SDL.h>
#include <OpenGL/gl3.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int SR = 44100;
static const float PI = 3.14159265f;

static float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static float smooth01(float x) { x = clampf(x, 0, 1); return x * x * (3 - 2 * x); }

static uint32_t rngState = 0x9E3779B9u;
static float frand() {
    rngState ^= rngState << 13; rngState ^= rngState >> 17; rngState ^= rngState << 5;
    return (rngState >> 8) * (1.0f / 16777216.0f);
}

static std::string shq(const std::string& s) {
    std::string o = "'";
    for (char c : s) { if (c == '\'') o += "'\\''"; else o += c; }
    return o + "'";
}

// ---------------------------------------------------------------- palettes

struct Palette {
    const char* name;
    float species[3][3]; // bass, mid, high
    float star[3];
    float tint[3];       // faint background wash
};

static const Palette PALETTES[] = {
    {"nebula",      {{1.00f, 0.22f, 0.55f}, {0.40f, 0.35f, 1.00f}, {0.30f, 0.90f, 1.00f}}, {1.00f, 0.75f, 0.50f}, {0.20f, 0.10f, 0.40f}},
    {"ember",       {{1.00f, 0.18f, 0.08f}, {1.00f, 0.55f, 0.12f}, {1.00f, 0.90f, 0.60f}}, {1.00f, 0.60f, 0.35f}, {0.35f, 0.08f, 0.10f}},
    {"aurora",      {{0.55f, 0.20f, 1.00f}, {0.10f, 0.90f, 0.60f}, {0.75f, 1.00f, 0.55f}}, {0.70f, 1.00f, 0.85f}, {0.05f, 0.25f, 0.30f}},
    {"ultraviolet", {{0.30f, 0.25f, 1.00f}, {0.85f, 0.25f, 1.00f}, {1.00f, 0.65f, 0.85f}}, {0.80f, 0.75f, 1.00f}, {0.15f, 0.05f, 0.35f}},
};
static const int NPALETTES = sizeof(PALETTES) / sizeof(PALETTES[0]);

// ---------------------------------------------------------------- audio analysis

static const int NBANDS = 5;
static const float BAND_HZ[NBANDS + 1] = {30, 120, 400, 2000, 6000, 16000};
static const int HOP = 512, NFFT = 2048;
static const float DROP_THRESHOLD = 0.22f;

struct Feat { float band[NBANDS]; float energy; };
struct Kick { double t; float strength; };

struct Track {
    std::string path, name;
    std::vector<float> pcm; // interleaved stereo
    double dur = 0;
    std::vector<Feat> feat; // one per HOP
    std::vector<Kick> kicks;
    std::vector<double> drops; // moments the track suddenly gets fuller
    int palette = 0;
    uint32_t seed = 1; // from the song name: each song gets its own constellation

    Feat at(double t) const {
        Feat z{};
        if (feat.empty()) return z;
        double f = t * SR / HOP;
        if (f <= 0) return feat.front();
        if (f >= feat.size() - 1) return feat.back();
        size_t i = (size_t)f; float a = (float)(f - i);
        for (int b = 0; b < NBANDS; b++) z.band[b] = feat[i].band[b] * (1 - a) + feat[i + 1].band[b] * a;
        z.energy = feat[i].energy * (1 - a) + feat[i + 1].energy * a;
        return z;
    }
};

static void fft(std::vector<std::complex<float>>& a) {
    size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        float ang = -2 * PI / len;
        std::complex<float> wl(cosf(ang), sinf(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1, 0);
            for (size_t j = 0; j < len / 2; j++) {
                auto u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v; a[i + j + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

static float percentile(std::vector<float> v, float p) {
    if (v.empty()) return 0;
    size_t k = std::min(v.size() - 1, (size_t)(p * v.size()));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// Onsets of whatever lives between loHz and hiHz in `pcm` (interleaved stereo):
// peaks of spectral flux that stand clear of their surroundings. On the low end
// of the full mix this catches every bass event; on one isolated stem it follows
// that part alone. `gate` drops hits weaker than that fraction of the source's
// typical strong hit: a stem's bleed and ghost notes would clutter the pulse.
static std::vector<Kick> detectOnsets(const std::vector<float>& pcm, float loHz, float hiHz, float gate = 0) {
    std::vector<Kick> out;
    long long nsamp = pcm.size() / 2;
    int nf = (int)(nsamp / HOP);
    if (nf < 3) return out;
    std::vector<float> window(NFFT);
    for (int i = 0; i < NFFT; i++) window[i] = 0.5f - 0.5f * cosf(2 * PI * i / (NFFT - 1));
    float binHz = (float)SR / NFFT;
    int lo = std::max(1, (int)(loHz / binHz)), hi = std::min(NFFT / 2, (int)(hiHz / binHz) + 1);
    std::vector<float> prevLog(hi, 0), flux(nf, 0);
    std::vector<std::complex<float>> buf(NFFT);
    for (int f = 0; f < nf; f++) {
        long long c = (long long)f * HOP - NFFT / 2;
        for (int i = 0; i < NFFT; i++) {
            long long sm = c + i;
            buf[i] = (sm >= 0 && sm < nsamp) ? 0.5f * (pcm[2 * sm] + pcm[2 * sm + 1]) * window[i] : 0;
        }
        fft(buf);
        for (int k = lo; k < hi; k++) {
            float l = logf(1 + 20 * std::abs(buf[k]));
            if (f > 0) flux[f] += std::max(0.0f, l - prevLog[k]);
            prevLog[k] = l;
        }
    }
    float fref = percentile(flux, 0.98f);
    if (fref < 0.05f) return out; // nothing audible in this band
    double fmean = 0;
    for (float x : flux) fmean += x;
    fmean /= nf;
    std::vector<double> fpre(nf + 1, 0);
    for (int f = 0; f < nf; f++) fpre[f + 1] = fpre[f] + flux[f];
    int win = (int)(0.5f * SR / HOP), last = -1000;
    for (int f = 1; f < nf - 1; f++) {
        int a = std::max(0, f - win), b = std::min(nf, f + win);
        float local = (float)((fpre[b] - fpre[a]) / (b - a));
        if (flux[f] < local * 1.7f + 0.5f * fmean || flux[f] < 0.2f * fref || f - last < 10) continue;
        bool peak = true;
        for (int k = std::max(0, f - 4); k <= std::min(nf - 1, f + 4); k++)
            if (flux[k] > flux[f]) peak = false;
        if (!peak) continue;
        out.push_back({(double)f * HOP / SR, flux[f]});
        last = f;
    }
    // Strength is relative to this source's own strong hits (the 85th percentile of its peaks).
    std::vector<float> peaks;
    for (const Kick& k : out) peaks.push_back(k.strength);
    float strong = std::max(1e-6f, percentile(peaks, 0.85f));
    out.erase(std::remove_if(out.begin(), out.end(), [&](const Kick& k) { return k.strength < gate * strong; }), out.end());
    for (Kick& k : out) k.strength = clampf(k.strength / strong, 0.3f, 1.2f);
    return out;
}

// Precompute per-hop band envelopes (normalised to each band's own loud level,
// so every instrument gets to drive something) and the drops.
static void analyze(Track& tr) {
    long long nsamp = tr.pcm.size() / 2;
    int nf = (int)(nsamp / HOP);
    if (nf < 2) return;
    std::vector<float> window(NFFT);
    for (int i = 0; i < NFFT; i++) window[i] = 0.5f - 0.5f * cosf(2 * PI * i / (NFFT - 1));

    std::vector<float> raw[NBANDS], rms(nf);
    for (auto& r : raw) r.resize(nf);
    std::vector<std::complex<float>> buf(NFFT);
    float binHz = (float)SR / NFFT;

    for (int f = 0; f < nf; f++) {
        long long c = (long long)f * HOP - NFFT / 2;
        double sq = 0;
        for (int i = 0; i < NFFT; i++) {
            long long s = c + i;
            float m = (s >= 0 && s < nsamp) ? 0.5f * (tr.pcm[2 * s] + tr.pcm[2 * s + 1]) : 0;
            sq += m * m;
            buf[i] = m * window[i];
        }
        rms[f] = sqrtf((float)(sq / NFFT));
        fft(buf);
        for (int b = 0; b < NBANDS; b++) {
            int lo = std::max(1, (int)(BAND_HZ[b] / binHz)), hi = std::min(NFFT / 2, (int)(BAND_HZ[b + 1] / binHz) + 1);
            double e = 0;
            for (int k = lo; k < hi; k++) e += std::norm(buf[k]);
            raw[b][f] = sqrtf((float)(e / std::max(1, hi - lo)));
        }
    }

    tr.feat.assign(nf, Feat{});
    float decay = expf(-(float)HOP / SR / 0.16f);
    for (int b = 0; b < NBANDS; b++) {
        float ref = std::max(1e-6f, percentile(raw[b], 0.95f));
        float env = 0;
        for (int f = 0; f < nf; f++) {
            float v = clampf(raw[b][f] / ref, 0, 1.6f);
            env = std::max(v, env * decay);
            tr.feat[f].band[b] = env;
        }
    }
    // Slow "how full is the track right now" curve: ~3 s moving average of loudness.
    float rref = std::max(1e-6f, percentile(rms, 0.95f));
    std::vector<double> pre(nf + 1, 0);
    for (int f = 0; f < nf; f++) pre[f + 1] = pre[f] + std::min(1.2f, rms[f] / rref);
    int half = (int)(1.5f * SR / HOP);
    for (int f = 0; f < nf; f++) {
        int a = std::max(0, f - half), b = std::min(nf, f + half);
        tr.feat[f].energy = clampf((float)((pre[b] - pre[a]) / (b - a)), 0, 1);
    }

    // Drops: where the two seconds ahead are much fuller (louder, more bass) than
    // the two behind. The camera and the ships stage their big moves on these.
    {
        float bref = std::max(1e-6f, percentile(raw[0], 0.95f));
        std::vector<double> full(nf + 1, 0);
        for (int f = 0; f < nf; f++)
            full[f + 1] = full[f] + 0.5f * std::min(1.2f, rms[f] / rref) + 0.5f * std::min(1.2f, raw[0][f] / bref);
        int w2 = (int)(2.0f * SR / HOP), lastDrop = -100000;
        std::vector<float> score(nf, 0);
        for (int f = w2; f < nf - w2; f++)
            score[f] = (float)((full[f + w2] - full[f]) - (full[f] - full[f - w2])) / w2;
        for (int f = w2; f < nf - w2; f++) {
            if (score[f] < DROP_THRESHOLD || f - lastDrop < 4 * w2) continue;
            bool peak = true;
            for (int k = f - w2; k <= f + w2; k++)
                if (score[k] > score[f]) peak = false;
            if (!peak) continue;
            tr.drops.push_back((double)f * HOP / SR);
            lastDrop = f;
        }
    }

}

static bool decode(const std::string& path, std::vector<float>& pcm) {
    pcm.clear();
    std::string cmd = "ffmpeg -v error -i " + shq(path) + " -f f32le -ac 2 -ar " + std::to_string(SR) + " -";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return false;
    float chunk[8192];
    size_t n;
    while ((n = fread(chunk, sizeof(float), 8192, p)) > 0) pcm.insert(pcm.end(), chunk, chunk + n);
    pclose(p);
    pcm.resize(pcm.size() & ~(size_t)1);
    return pcm.size() >= (size_t)SR;
}
static bool fileExists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f) fclose(f);
    return f != nullptr;
}

// What the centre star pulses to. Each choice is one or two separated stems (see
// `make stems`) and the band of them to listen to; "kick" is the drums stem's
// low end, which leaves the hi-hats and snare out.
// `gate`: only hits at least this strong, relative to the source's typical strong
// hit, are kept. Melodic parts get a lower gate, since notes vary in loudness.
struct PulseSource { const char* name; const char* stem; const char* stem2; float loHz, hiHz, gate; };
static const PulseSource PULSES[] = {
    // The synth line: the low register of every pitched part, drums excluded.
    // This is what the low end of the full mix catches, minus the kick.
    {"synth", "bass", "other", 20, 160, 0.2f},
    // Note onsets higher up. Demucs has no lead stem; synths land in "other" and
    // voice-like leads in "vocals", so this listens to both above the bass.
    {"melody", "other", "vocals", 180, 6000, 0.4f},
    {"kick", "drums", nullptr, 20, 160, 0.55f},   {"drums", "drums", nullptr, 20, 8000, 0.55f},
    {"hats", "drums", nullptr, 6000, 16000, 0.55f}, {"bass", "bass", nullptr, 20, 400, 0.55f},
    {"other", "other", nullptr, 80, 6000, 0.4f},   {"vocals", "vocals", nullptr, 80, 6000, 0.4f},
};

// Onsets for one pulse source: a name from PULSES, "mix" for the low end of the
// full mix, a path to any audio file aligned with the song, or several of these
// joined with '+' (their hits are merged). False if its audio cannot be read.
static bool pulseHits(const Track& tr, const std::string& want, std::vector<Kick>& hits, std::string& label) {
    size_t plus = want.find('+');
    if (plus != std::string::npos) {
        std::vector<Kick> a, b;
        std::string la, lb;
        if (!pulseHits(tr, want.substr(0, plus), a, la) || !pulseHits(tr, want.substr(plus + 1), b, lb)) return false;
        a.insert(a.end(), b.begin(), b.end());
        std::sort(a.begin(), a.end(), [](const Kick& x, const Kick& y) { return x.t < y.t; });
        // Two sources hitting together make one pulse, the stronger of the two.
        hits.clear();
        for (const Kick& k : a) {
            if (!hits.empty() && k.t - hits.back().t < 0.09) hits.back().strength = std::max(hits.back().strength, k.strength);
            else hits.push_back(k);
        }
        label = la + " + " + lb;
        return true;
    }
    if (want == "mix") { hits = detectOnsets(tr.pcm, 20, 160); label = "low end of the full mix"; return true; }
    size_t slash = tr.path.find_last_of('/'), dot = tr.name.find_last_of('.');
    std::string stemDir = (slash == std::string::npos ? "" : tr.path.substr(0, slash + 1)) + "stems/htdemucs/" + tr.name.substr(0, dot) + "/";
    std::string file = want, file2;
    float lo = 20, hi = 8000, gate = 0.55f;
    label = want;
    for (const PulseSource& ps : PULSES)
        if (want == ps.name) {
            file = stemDir + ps.stem + ".wav";
            if (ps.stem2) file2 = stemDir + ps.stem2 + ".wav";
            lo = ps.loHz; hi = ps.hiHz; gate = ps.gate;
            label = want + " (" + ps.stem + (ps.stem2 ? std::string(" + ") + ps.stem2 : "") + " stem" + (ps.stem2 ? "s)" : ")");
        }
    std::vector<float> pcm, more;
    if (!fileExists(file) || !decode(file, pcm)) return false;
    if (!file2.empty() && fileExists(file2) && decode(file2, more))
        for (size_t k = 0; k < std::min(pcm.size(), more.size()); k++) pcm[k] += more[k];
    hits = detectOnsets(pcm, lo, hi, gate);
    return true;
}

// Sets what the centre star pulses to. An empty `pulse` means "the synth line if
// this song has stems, otherwise the mix".
static void choosePulse(Track& tr, const std::string& pulse) {
    std::string label;
    if (!pulseHits(tr, pulse.empty() ? "synth" : pulse, tr.kicks, label) || tr.kicks.empty()) {
        if (!pulse.empty()) fprintf(stderr, "  pulse: nothing usable from '%s'; using the full mix\n", pulse.c_str());
        pulseHits(tr, "mix", tr.kicks, label);
    }
    printf("  pulse: %s, %zu hits\n", label.c_str(), tr.kicks.size());
}

// For --pulses: how many hits each source gives in this song and how steady
// they are (share of gaps that fall on a grid of half the commonest gap).
static void comparePulses(const Track& tr) {
    printf("  %-12s %6s %10s %9s\n", "source", "hits", "per min", "steady");
    std::vector<Kick> mix;
    std::string unused;
    pulseHits(tr, "mix", mix, unused);
    for (const char* name : {"mix", "synth", "melody", "kick", "drums", "hats", "bass", "other", "vocals"}) {
        std::vector<Kick> hits;
        std::string label;
        if (!pulseHits(tr, name, hits, label) || hits.size() < 3) { printf("  %-12s %6zu %10s %9s\n", name, hits.size(), "-", "-"); continue; }
        std::vector<double> gaps;
        for (size_t i = 1; i < hits.size(); i++) gaps.push_back(hits[i].t - hits[i - 1].t);
        int bins[100] = {0}, top = 0; // 20 ms bins up to 2 s
        for (double g : gaps) bins[std::min(99, (int)(g / 0.02))]++;
        for (int b = 1; b < 100; b++)
            if (bins[b] > bins[top]) top = b;
        double half = (top + 0.5) * 0.02 / 2;
        int steady = 0;
        for (double g : gaps) {
            double m = g / half;
            if (fabs(m - round(m)) * half < std::min(0.03, 0.2 * half)) steady++;
        }
        // How much of what the full mix's low end catches does this source also catch,
        // and how much of this source is something the mix did not have?
        int caught = 0, extra = 0;
        for (const Kick& m : mix)
            for (const Kick& k : hits)
                if (fabs(k.t - m.t) < 0.05) { caught++; break; }
        for (const Kick& k : hits) {
            bool found = false;
            for (const Kick& m : mix)
                if (fabs(k.t - m.t) < 0.05) found = true;
            extra += !found;
        }
        printf("  %-12s %6zu %10.0f %8.0f%%   gap %.2fs   catches %3.0f%% of mix hits, %3.0f%% of its own are new\n", name,
               hits.size(), hits.size() / tr.dur * 60, 100.0 * steady / gaps.size(), (top + 0.5) * 0.02,
               100.0 * caught / std::max<size_t>(1, mix.size()), 100.0 * extra / hits.size());
    }
}

static bool loadTrack(const std::string& path, Track& tr, int paletteOverride, const std::string& pulse) {
    tr = Track();
    tr.path = path;
    size_t slash = path.find_last_of('/');
    tr.name = slash == std::string::npos ? path : path.substr(slash + 1);
    if (!decode(path, tr.pcm)) { fprintf(stderr, "could not decode %s\n", path.c_str()); return false; }
    tr.dur = (double)(tr.pcm.size() / 2) / SR;
    analyze(tr);
    uint32_t h = 2166136261u;
    for (char c : tr.name) h = (h ^ (uint8_t)c) * 16777619u;
    tr.seed = h | 1;
    tr.palette = paletteOverride >= 0 ? paletteOverride % NPALETTES : (int)(h % NPALETTES);

    printf("%s: %.1fs, palette '%s', drops at", tr.name.c_str(), tr.dur, PALETTES[tr.palette].name);
    for (double d : tr.drops) printf(" %.1fs", d);
    printf(tr.drops.empty() ? " (none found)\n" : "\n");
    choosePulse(tr, pulse);
    return true;
}

// ---------------------------------------------------------------- simulation

struct Vec3 { float x = 0, y = 0, z = 0; };
static Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static float length(Vec3 a) { return sqrtf(dot(a, a)); }
static Vec3 normalize(Vec3 a) { return a * (1.0f / (length(a) + 1e-9f)); }
static Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }

struct Wave { float t0, strength, x, y; }; // a ring spreading from (x, y)
static const int MAXWAVES = 8;
static const float WAVE_SPEED = 0.95f, WAVE_LIFE = 1.4f, WAVE_WIDTH = 0.07f;

// Stars ride fixed slow orbits around the core; each one listens to one band.
// Planets circle their star, moons circle their planet, and both spin up as
// that band gets louder. Everything lives in the galactic plane z = 0.
struct Moon { float r, phase, size, seed; };
enum PlanetType { ROCK, DESERT, OCEAN, LAVA, GAS, ICE };
struct Planet {
    float r, phase, size, seed;
    float extent; // radius of everything travelling with it: moons, rings
    float col[3];
    PlanetType type;
    bool ringed;
    std::vector<Moon> moons;
};
struct Star {
    float bx, by, size; // bx, by: place in the constellation before it turns
    float radius = 0; // outer edge of its planetary system
    int band, species;
    float x = 0, y = 0, level = 0;
    float glow = 0; // level smoothed over about half a second, for brightness
    float pace = 0; // level averaged over a couple of seconds, for orbital speed
    std::vector<Planet> planets;
};
static const int NSTARS = 6;

static void starColor(const Star& st, const Palette& pal, float out[3]) {
    float white = st.species == 2 ? 0.45f : 0.2f; // the high stars burn hotter
    for (int k = 0; k < 3; k++) out[k] = pal.species[st.species][k] * (1 - white) + white;
}
// A planet's own colour, leaning a little toward its star's so each system reads as one family.
static void planetColor(const Star& st, const Planet& pl, const Palette& pal, float out[3]) {
    float tint = pl.type == LAVA ? 0.0f : 0.2f;
    for (int k = 0; k < 3; k++) out[k] = pl.col[k] * (1 - tint) + pal.species[st.species][k] * tint;
}
static Vec3 planetPos(const Star& st, const Planet& pl) {
    return {st.x + pl.r * cosf(pl.phase), st.y + pl.r * sinf(pl.phase), 0};
}

// Ships fly in squadrons from star to star, one species per slice of the
// spectrum, each dragging a ribbon of its recent positions.
static const int TRAIL = 48;
static const float TRAIL_DT = 0.06f;
struct Ship {
    float x, y, z, vx, vy, phase, rank;
    int s, squad;
    float trail[TRAIL][3];
};
struct Squad { int species, dest; float cx, cy, hx, hy; };
static const int NSQUADS = 6;

// The camera flies a loop of shots, easing from each one into the next.
enum ShotType { WIDE, SYSTEM, PLANET, CHASE };
struct Shot { ShotType type = WIDE; int a = 0, b = 0; float t0 = 0, ang0 = 0, dur = 9, trans = 6, arc = 0.14f; };
struct Pose { Vec3 pos, tgt; };
// A drop gets a faster, higher swoop that starts DROP_LEAD seconds early, so the
// camera is already accelerating as it hits.
static const float DROP_LEAD = 1.2f;
static const float STAR_BERTH = 7.0f; // camera stays this many star radii from any sun

struct Sim {
    // World is [-1,1]^2 in the plane. Fluid is periodic.
    float hx = 1, hy = 1, h = 0;
    int nx = 0, ny = 0;
    std::vector<float> u, v, u0, v0, p, p1, div, curl, swirlFalloff, dye[3], dye0[3];
    std::vector<int> xm, xp, ym, yp; // wrapped neighbour indices
    std::vector<Ship> ships;
    Squad squads[NSQUADS];
    std::vector<Star> stars;
    std::vector<Wave> waves;
    float time = 0, trailClock = 0, turn = 0;
    float slv[3] = {0, 0, 0}; // band levels smoothed for brightness, so lights swell rather than flicker
    float pulseDye = 0;       // kick dye waiting to be bled in around the star that pulsed
    float pulseX = 0, pulseY = 0;
    int lanes[NSTARS - 1][2]; // hyperlanes: a spanning tree over the stars
    float surge = 0;          // 1 at a drop, decaying: ships fly faster and turn harder

    Shot shot, prevShot;
    int shotCount = 0, lastStar = 0;
    bool ending = false, dropArmed = true;
    Pose cam;
    Vec3 camFwd{1, 0, 0}; // where the camera actually looks: eased toward cam.tgt, turn rate capped

    int ix(int i, int j) const {
        if ((unsigned)i >= (unsigned)nx) { i %= nx; if (i < 0) i += nx; }
        if ((unsigned)j >= (unsigned)ny) { j %= ny; if (j < 0) j += ny; }
        return i + j * nx;
    }

    void init(int nships) {
        nx = ny = 192;
        h = 2 * hx / nx;
        size_t n = (size_t)nx * ny;
        for (auto* f : {&u, &v, &u0, &v0, &p, &p1, &div, &curl}) f->assign(n, 0);
        for (int c = 0; c < 3; c++) { dye[c].assign(n, 0); dye0[c].assign(n, 0); }
        xm.resize(nx); xp.resize(nx); ym.resize(ny); yp.resize(ny);
        for (int i = 0; i < nx; i++) { xm[i] = (i + nx - 1) % nx; xp[i] = (i + 1) % nx; }
        for (int j = 0; j < ny; j++) { ym[j] = (j + ny - 1) % ny; yp[j] = (j + 1) % ny; }
        swirlFalloff.resize(n);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                float x = (i + 0.5f) * h - hx, y = (j + 0.5f) * h - hy;
                swirlFalloff[i + j * nx] = expf(-(x * x + y * y) / 0.9f);
            }
        waves.clear();

        // One star at the heart of the constellation: the bass star, the source of
        // every pulse. The rest are scattered around it, anywhere inside a tilted
        // ellipse as long as each keeps its distance from the others, so they form
        // no ring. The whole pattern turns slowly about the centre star as one.
        static const int BAND[NSTARS] = {0, 2, 3, 1, 4, 0};
        static const float STAR_SIZE[NBANDS] = {0.034f, 0.029f, 0.025f, 0.022f, 0.019f};
        float apart = 0.46f, tilt = frand() * PI;
        for (;;) {
            stars.clear();
            for (int tries = 0; (int)stars.size() < NSTARS && tries < 300; tries++) {
                float a = frand() * 2 * PI, r = sqrtf(frand());
                float ex = 0.92f * r * cosf(a), ey = 0.62f * r * sinf(a);
                float x = ex * cosf(tilt) - ey * sinf(tilt), y = ex * sinf(tilt) + ey * cosf(tilt);
                if (stars.empty()) x = y = 0;
                bool clear = true;
                for (const Star& o : stars)
                    if (hypotf(o.bx - x, o.by - y) < apart) clear = false;
                if (!clear) continue;
                Star st;
                st.bx = x; st.by = y;
                st.band = BAND[stars.size()];
                st.species = st.band == 0 ? 0 : (st.band <= 2 ? 1 : 2);
                st.size = stars.empty() ? 0.038f : STAR_SIZE[st.band];
                stars.push_back(st);
            }
            if ((int)stars.size() == NSTARS) break;
            apart -= 0.005f; // too tight to fit: ease off and try again
        }
        for (Star& st : stars) placeStar(st);
        for (int i = 0; i < NSTARS; i++) {
            float nearest = 1e9f;
            for (int j = 0; j < NSTARS; j++)
                if (j != i) nearest = std::min(nearest, hypotf(stars[i].x - stars[j].x, stars[i].y - stars[j].y));
            buildSystem(stars[i], std::min(0.27f, nearest / 2 - 0.02f));
        }
        // Hyperlanes: the minimum spanning tree, so every star is reachable and
        // no lane closes a loop.
        {
            bool in[NSTARS] = {true};
            for (int e = 0; e < NSTARS - 1; e++) {
                int ba = 0, bb = 0;
                float best = 1e9f;
                for (int a = 0; a < NSTARS; a++)
                    for (int b = 0; b < NSTARS; b++) {
                        if (!in[a] || in[b]) continue;
                        float d = hypotf(stars[a].bx - stars[b].bx, stars[a].by - stars[b].by);
                        if (d < best) { best = d; ba = a; bb = b; }
                    }
                in[bb] = true;
                lanes[e][0] = ba; lanes[e][1] = bb;
            }
        }

        // Squadrons start out circling a star, already bound for another.
        nships = std::max(NSQUADS, nships);
        ships.clear();
        int per = nships / NSQUADS;
        for (int q = 0; q < NSQUADS; q++) {
            const Star& home = stars[q % NSTARS];
            float th = frand() * 2 * PI;
            squads[q] = {q % 3, (q + 2) % NSTARS, home.x, home.y, -sinf(th), cosf(th)};
            for (int i = 0; i < per; i++) {
                Ship sh;
                float r = home.radius * 0.8f + 0.04f * frand(), a = th + 0.5f * (frand() - 0.5f);
                sh.x = home.x + r * cosf(a); sh.y = home.y + r * sinf(a); sh.z = 0.04f;
                sh.vx = -sinf(a) * 0.12f; sh.vy = cosf(a) * 0.12f;
                sh.phase = frand() * 2 * PI; sh.rank = (float)i / per;
                sh.s = q % 3; sh.squad = q;
                for (auto& t : sh.trail) { t[0] = sh.x; t[1] = sh.y; t[2] = sh.z; }
                ships.push_back(sh);
            }
        }
        startCamera();
    }

    // Lays out one planetary system inside `limit`: rocky worlds close in, giants
    // further out, each orbit spaced by what actually travels on it and its
    // neighbour, so nothing ever crosses anything else.
    void buildSystem(Star& st, float limit) {
        static const float COLORS[6][3][3] = {
            {{0.56f, 0.52f, 0.48f}, {0.48f, 0.44f, 0.42f}, {0.62f, 0.55f, 0.46f}}, // rock
            {{0.82f, 0.46f, 0.26f}, {0.78f, 0.60f, 0.36f}, {0.66f, 0.36f, 0.26f}}, // desert
            {{0.10f, 0.30f, 0.70f}, {0.08f, 0.40f, 0.62f}, {0.14f, 0.24f, 0.60f}}, // ocean
            {{0.10f, 0.07f, 0.07f}, {0.13f, 0.08f, 0.06f}, {0.08f, 0.07f, 0.09f}}, // lava crust
            {{0.86f, 0.70f, 0.50f}, {0.80f, 0.55f, 0.40f}, {0.74f, 0.52f, 0.62f}}, // gas
            {{0.45f, 0.76f, 0.86f}, {0.32f, 0.50f, 0.90f}, {0.55f, 0.86f, 0.74f}}, // ice
        };
        float inner = st.size * 1.9f;
        float phase = frand() * 2 * PI;
        for (int k = 0; k < 6; k++) {
            float where = (inner - st.size) / std::max(0.01f, limit - st.size); // 0 at the star, 1 at the edge
            Planet pl;
            // A giant that will not fit gets one retry as a small bare rock.
            for (int attempt = 0; attempt < 2; attempt++) {
                float pick = frand();
                if (attempt == 1) pl.type = ROCK;
                else if (where < 0.3f) pl.type = pick < 0.45f ? LAVA : (pick < 0.8f ? ROCK : DESERT);
                else if (where < 0.5f) pl.type = pick < 0.4f ? OCEAN : (pick < 0.75f ? DESERT : ROCK);
                else pl.type = pick < 0.55f ? GAS : ICE;
                bool giant = pl.type == GAS || pl.type == ICE;
                pl.size = pl.type == GAS ? 0.015f + 0.005f * frand() : pl.type == ICE ? 0.012f + 0.004f * frand()
                                                                                     : 0.007f + 0.005f * frand();
                pl.ringed = giant && frand() < 0.5f;
                pl.seed = frand() * 0.998f;
                pl.moons.clear();
                pl.extent = pl.ringed ? pl.size * 2.3f : pl.size;
                int nm = attempt == 1 ? 0 : (int)(frand() * (giant ? 3.99f : 1.99f));
                float mr = std::max(pl.size * 1.8f, pl.extent + 0.004f);
                for (int m = 0; m < nm; m++) {
                    Moon mo;
                    mo.size = std::min(0.0045f, pl.size * (0.18f + 0.12f * frand()));
                    mo.r = mr + mo.size;
                    mo.phase = frand() * 2 * PI;
                    mo.seed = frand() * 0.998f;
                    pl.moons.push_back(mo);
                    pl.extent = mo.r + mo.size;
                    mr = pl.extent + 0.004f;
                }
                if (inner + 2 * pl.extent + 0.008f <= limit) break;
                // A giant that is merely crowded by its moons and rings sheds them and stays.
                if (giant && inner + 2 * pl.size + 0.008f <= limit) { pl.moons.clear(); pl.ringed = false; pl.extent = pl.size; break; }
                pl.size = 0;
            }
            if (pl.size == 0) break;
            pl.r = inner + pl.extent + 0.008f;
            phase += 2.4f; // golden angle: successive planets start well apart
            pl.phase = phase;
            memcpy(pl.col, COLORS[pl.type][(int)(frand() * 2.99f)], sizeof pl.col);
            st.planets.push_back(pl);
            inner = pl.r + pl.extent + 0.008f + 0.014f * frand();
        }
        st.radius = inner;
    }

    void placeStar(Star& st) const {
        st.x = st.bx * cosf(turn) - st.by * sinf(turn);
        st.y = st.bx * sinf(turn) + st.by * cosf(turn);
    }
    void moveStars(float dt, const Feat& f) {
        turn += 0.011f * dt * (0.5f + f.energy);
        for (Star& st : stars) {
            st.level = f.band[st.band];
            st.glow += (st.level - st.glow) * (1 - expf(-dt / 0.6f));
            st.pace += (st.level - st.pace) * (1 - expf(-dt / 2.5f));
            placeStar(st);
            // Orbits follow the slow average, never the raw level: a planet's speed
            // may drift with the music but must not twitch with each note.
            for (Planet& pl : st.planets) {
                pl.phase += dt * 0.55f * powf(0.08f / pl.r, 1.5f) * (0.5f + 1.2f * st.pace);
                for (Moon& mo : pl.moons) mo.phase += dt * 0.9f * (0.6f + 1.1f * st.pace) * powf(pl.size * 2.5f / mo.r, 1.5f);
            }
        }
    }
    // Each star breathes dye and an outward wind into the fluid, as loud as its band.
    void stellarWind(float dt, const Palette& pal) {
        const float R = 0.045f;
        int r = (int)(2.4f * R / h);
        for (const Star& st : stars) {
            float col[3];
            starColor(st, pal, col);
            int ci = (int)lroundf(gx(st.x)), cj = (int)lroundf(gy(st.y));
            for (int dj = -r; dj <= r; dj++)
                for (int di = -r; di <= r; di++) {
                    float x = (ci + di + 0.5f) * h - hx - st.x, y = (cj + dj + 0.5f) * h - hy - st.y;
                    float q = (x * x + y * y) / (R * R), fall = expf(-q);
                    int c = ix(ci + di, cj + dj);
                    for (int k = 0; k < 3; k++) dye[k][c] += col[k] * dt * 1.0f * st.level * fall;
                    u[c] += x / R * fall * dt * 1.6f * st.level;
                    v[c] += y / R * fall * dt * 1.6f * st.level;
                }
        }
    }

    float bilerp(const std::vector<float>& f, float gx, float gy) const {
        int i = (int)floorf(gx), j = (int)floorf(gy);
        float a = gx - i, b = gy - j;
        return (f[ix(i, j)] * (1 - a) + f[ix(i + 1, j)] * a) * (1 - b) +
               (f[ix(i, j + 1)] * (1 - a) + f[ix(i + 1, j + 1)] * a) * b;
    }
    void splat(std::vector<float>& f, float gx, float gy, float amt) {
        int i = (int)floorf(gx), j = (int)floorf(gy);
        float a = gx - i, b = gy - j;
        f[ix(i, j)] += amt * (1 - a) * (1 - b); f[ix(i + 1, j)] += amt * a * (1 - b);
        f[ix(i, j + 1)] += amt * (1 - a) * b;   f[ix(i + 1, j + 1)] += amt * a * b;
    }
    float gx(float x) const { return (x + hx) / h - 0.5f; }
    float gy(float y) const { return (y + hy) / h - 0.5f; }

    void advect(std::vector<float>& dst, const std::vector<float>& src, float dt) {
        float s = dt / h;
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int c = i + j * nx;
                dst[c] = bilerp(src, i - s * u0[c], j - s * v0[c]);
            }
    }

    // Outward acceleration at (x, y) from every live wave front.
    void wavePush(float x, float y, float& ax, float& ay) const {
        for (const Wave& w : waves) {
            float age = time - w.t0, dx = x - w.x, dy = y - w.y, d = sqrtf(dx * dx + dy * dy) + 1e-4f;
            float q = (d - (0.05f + WAVE_SPEED * age)) / WAVE_WIDTH;
            if (q * q > 12) continue;
            float a = w.strength * w.strength * expf(-2.6f * age) * expf(-q * q) / d;
            ax += dx * a; ay += dy * a;
        }
    }

    // A low-end hit: the centre star sends out a ring.
    void kick(float strength) {
        const Star& src = stars[0];
        if ((int)waves.size() >= MAXWAVES) waves.erase(waves.begin());
        // Compressed, so a quiet note still makes a ring worth seeing.
        float punch = 0.6f + 0.4f * strength;
        waves.push_back({time, punch, src.x, src.y});
        pulseDye += 0.04f * punch * punch;
        pulseX = src.x; pulseY = src.y;
    }
    // The kick's dye is bled in over a few frames rather than stamped at once:
    // nothing on screen may switch on in one frame.
    void feedPulse(float dt, const Palette& pal) {
        float amt = pulseDye * (1 - expf(-dt / 0.15f));
        pulseDye -= amt;
        int r = (int)(0.22f / h), ci = (int)lroundf(gx(pulseX)), cj = (int)lroundf(gy(pulseY));
        for (int dj = -r; dj <= r; dj++)
            for (int di = -r; di <= r; di++) {
                float d = sqrtf((float)(di * di + dj * dj)) * h;
                float a = amt * expf(-(d * d) / (0.09f * 0.09f));
                int c = ix(ci + di, cj + dj);
                for (int k = 0; k < 3; k++) dye[k][c] += pal.star[k] * a;
            }
    }

    void stepFluid(float dt, const float lv[3]) {
        // Forces: galactic swirl (mids) + shockwaves.
        float swirl = 0.5f * (0.25f + lv[1]);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                float x = (i + 0.5f) * h - hx, y = (j + 0.5f) * h - hy;
                int c = i + j * nx;
                float sw = swirl * swirlFalloff[c], wx = 0, wy = 0;
                wavePush(x, y, wx, wy);
                u[c] += dt * (-y * sw + 3.0f * wx);
                v[c] += dt * (x * sw + 3.0f * wy);
            }
        // Vorticity confinement keeps the small curls alive.
        float i2h = 1 / (2 * h);
        for (int j = 0; j < ny; j++) {
            const float *vr = &v[j * nx], *uu = &u[yp[j] * nx], *ud = &u[ym[j] * nx];
            for (int i = 0; i < nx; i++) curl[i + j * nx] = (vr[xp[i]] - vr[xm[i]] - uu[i] + ud[i]) * i2h;
        }
        const float eps = 9.0f;
        for (int j = 0; j < ny; j++) {
            const float *cr = &curl[j * nx], *cu = &curl[yp[j] * nx], *cd = &curl[ym[j] * nx];
            for (int i = 0; i < nx; i++) {
                float gxm = fabsf(cr[xp[i]]) - fabsf(cr[xm[i]]), gym = fabsf(cu[i]) - fabsf(cd[i]);
                float k = dt * eps * h * cr[i] / (sqrtf(gxm * gxm + gym * gym) + 1e-5f);
                u[i + j * nx] += k * gym;
                v[i + j * nx] -= k * gxm;
            }
        }

        u0 = u; v0 = v;
        advect(u, u0, dt); advect(v, v0, dt);

        // Projection (periodic Gauss-Seidel).
        for (int j = 0; j < ny; j++) {
            const float *ur = &u[j * nx], *vu = &v[yp[j] * nx], *vd = &v[ym[j] * nx];
            for (int i = 0; i < nx; i++) {
                div[i + j * nx] = (ur[xp[i]] - ur[xm[i]] + vu[i] - vd[i]) * i2h * h * h;
                p[i + j * nx] = 0;
            }
        }
        // Jacobi rather than Gauss-Seidel: no loop-carried dependency, so the
        // inner loop vectorises.
        for (int it = 0; it < 30; it++) {
            for (int j = 0; j < ny; j++) {
                const float *pr = &p[j * nx], *pu = &p[yp[j] * nx], *pd = &p[ym[j] * nx], *dr = &div[j * nx];
                float* pn = &p1[j * nx];
                pn[0] = (pr[1] + pr[nx - 1] + pu[0] + pd[0] - dr[0]) * 0.25f;
                for (int i = 1; i < nx - 1; i++) pn[i] = (pr[i + 1] + pr[i - 1] + pu[i] + pd[i] - dr[i]) * 0.25f;
                pn[nx - 1] = (pr[0] + pr[nx - 2] + pu[nx - 1] + pd[nx - 1] - dr[nx - 1]) * 0.25f;
            }
            p.swap(p1);
        }
        float vdamp = expf(-0.35f * dt);
        for (int j = 0; j < ny; j++) {
            const float *pr = &p[j * nx], *pu = &p[yp[j] * nx], *pd = &p[ym[j] * nx];
            for (int i = 0; i < nx; i++) {
                int c = i + j * nx;
                u[c] = (u[c] - (pr[xp[i]] - pr[xm[i]]) * i2h) * vdamp;
                v[c] = (v[c] - (pu[i] - pd[i]) * i2h) * vdamp;
            }
        }

        u0 = u; v0 = v;
        float ddamp = expf(-0.7f * dt);
        for (int k = 0; k < 3; k++) {
            dye0[k].swap(dye[k]);
            advect(dye[k], dye0[k], dt);
            for (float& d : dye[k]) d *= ddamp;
        }
    }

    // Quiet passages thin the squadrons: ships past the "active" rank go dark.
    static float fadeOf(const Ship& s, float active) { return smooth01((active - s.rank) / 0.12f + 1); }

    // Next stop: a star a real journey away, so the squadron sets off on a long
    // arc instead of dithering between neighbours.
    void redirect(Squad& sq) {
        int next = sq.dest, tries = 0;
        float farthest = 0;
        for (int i = 0; i < NSTARS; i++) {
            float dd = hypotf(stars[i].x - sq.cx, stars[i].y - sq.cy);
            if (dd > farthest) { farthest = dd; next = i; }
        }
        while (tries++ < 8) {
            int c = (int)(frand() * (NSTARS - 0.01f));
            if (c != sq.dest && hypotf(stars[c].x - sq.cx, stars[c].y - sq.cy) > 0.55f) { next = c; break; }
        }
        sq.dest = next;
    }
    // The track just got fuller: every squadron surges and wheels toward a new star.
    void drop() {
        surge = 1;
        for (Squad& sq : squads) redirect(sq);
    }

    void stepShips(float dt, const float lv[3], float energy, const Palette& pal) {
        static const float SPEED[3] = {0.10f, 0.135f, 0.17f};
        static const float DEPOSIT[3] = {4.0f, 3.0f, 2.0f};

        float sx[NSQUADS] = {0}, sy[NSQUADS] = {0}, svx[NSQUADS] = {0}, svy[NSQUADS] = {0};
        int cnt[NSQUADS] = {0};
        for (const Ship& s : ships) { sx[s.squad] += s.x; sy[s.squad] += s.y; svx[s.squad] += s.vx; svy[s.squad] += s.vy; cnt[s.squad]++; }
        for (int q = 0; q < NSQUADS; q++) {
            Squad& sq = squads[q];
            if (!cnt[q]) continue;
            sq.cx = sx[q] / cnt[q]; sq.cy = sy[q] / cnt[q];
            svx[q] /= cnt[q]; svy[q] /= cnt[q];
            // Heading is low-passed: the chase camera steers by it.
            float sp = sqrtf(svx[q] * svx[q] + svy[q] * svy[q]) + 1e-6f, k = 1 - expf(-dt / 1.5f);
            sq.hx += (svx[q] / sp - sq.hx) * k; sq.hy += (svy[q] / sp - sq.hy) * k;
            const Star& d = stars[sq.dest];
            if (hypotf(d.x - sq.cx, d.y - sq.cy) < d.radius + 0.12f) redirect(sq);
        }

        float active = 0.3f + 0.7f * energy;
        size_t n = ships.size();
        std::vector<float> ax(n), ay(n);
        for (size_t i = 0; i < n; i++) {
            const Ship& b = ships[i];
            const Squad& sq = squads[b.squad];
            float fx = 0, fy = 0;
            for (size_t j = 0; j < n; j++) {
                if (j == i) continue;
                float dx = b.x - ships[j].x, dy = b.y - ships[j].y, d = sqrtf(dx * dx + dy * dy) + 1e-6f;
                if (d < 0.034f) { float w = 1.2f * (1 - d / 0.034f) / d; fx += dx * w; fy += dy * w; }
            }
            fx += (sq.cx - b.x) * 0.5f + (svx[b.squad] - b.vx) * 2.2f;
            fy += (sq.cy - b.y) * 0.5f + (svy[b.squad] - b.vy) * 2.2f;
            // Aim to one side of the destination star, to swing past rather than at it.
            const Star& dst = stars[sq.dest];
            float tx = dst.x - b.x, ty = dst.y - b.y, td = sqrtf(tx * tx + ty * ty) + 1e-6f;
            float side = (b.squad & 1 ? 1 : -1) * dst.radius * 0.8f;
            float aimx = dst.x - ty / td * side - b.x, aimy = dst.y + tx / td * side - b.y, ad = sqrtf(aimx * aimx + aimy * aimy) + 1e-6f;
            float cruise = SPEED[b.s] * (0.65f + 0.7f * slv[b.s]) * (1 + 0.9f * surge);
            fx += (aimx / ad * cruise - b.vx) * 0.9f; fy += (aimy / ad * cruise - b.vy) * 0.9f;
            for (const Star& st : stars) {
                float dx = b.x - st.x, dy = b.y - st.y, d = sqrtf(dx * dx + dy * dy) + 1e-6f, keep = st.size * 5.5f;
                if (d < keep) { float w = 1.4f * (1 - d / keep) / d; fx += dx * w; fy += dy * w; }
            }
            float wx = 0, wy = 0;
            wavePush(b.x, b.y, wx, wy);
            fx += 0.3f * wx; fy += 0.3f * wy;
            // Gentle turns only.
            float f = sqrtf(fx * fx + fy * fy), maxF = 0.3f * (1 + 1.5f * surge);
            if (f > maxF) { fx *= maxF / f; fy *= maxF / f; }
            float sp = sqrtf(b.vx * b.vx + b.vy * b.vy) + 1e-5f;
            ax[i] = fx + b.vx / sp * (cruise - sp) * 2.0f;
            ay[i] = fy + b.vy / sp * (cruise - sp) * 2.0f;
        }

        trailClock += dt;
        bool sample = trailClock >= TRAIL_DT;
        if (sample) trailClock = fmodf(trailClock, TRAIL_DT);
        for (size_t i = 0; i < n; i++) {
            Ship& b = ships[i];
            b.vx += ax[i] * dt; b.vy += ay[i] * dt;
            b.x += b.vx * dt; b.y += b.vy * dt;
            b.z = 0.032f + 0.012f * (b.squad % 3) + 0.008f * sinf(b.phase + time * 0.3f);
            if (sample) {
                memmove(b.trail[1], b.trail[0], sizeof(float) * 3 * (TRAIL - 1));
                b.trail[0][0] = b.x; b.trail[0][1] = b.y; b.trail[0][2] = b.z;
            }
            // Exhaust: a little dye and a nudge to the fluid.
            float fade = fadeOf(b, active), g_x = gx(b.x), g_y = gy(b.y);
            float amt = dt * fade * DEPOSIT[b.s] * (0.15f + lv[b.s]);
            for (int k = 0; k < 3; k++) splat(dye[k], g_x, g_y, pal.species[b.s][k] * amt);
            splat(u, g_x, g_y, b.vx * dt * fade * 6.0f); splat(v, g_x, g_y, b.vy * dt * fade * 6.0f);
        }
    }

    // ---- camera

    // Portrait frames ride higher and look down more steeply, so the orbital
    // plane fills the height of the frame instead of a band across its middle.
    float lift = 1;
    Pose poseOf(const Shot& sh) const {
        Pose p = flatPose(sh);
        p.pos.z *= lift;
        return p;
    }

    Pose flatPose(const Shot& sh) const {
        float t = time - sh.t0;
        switch (sh.type) {
        case SYSTEM: {
            const Star& st = stars[sh.a];
            float th = sh.ang0 + 0.07f * t, dist = st.radius * 1.15f + 0.10f;
            return {{st.x + dist * cosf(th), st.y + dist * sinf(th), dist * (0.30f + 0.06f * sinf(0.2f * t))}, {st.x, st.y, 0}};
        }
        case PLANET: {
            // Hang off the planet's flank, where the terminator shows, drifting slowly round it.
            const Star& st = stars[sh.a];
            const Planet& pl = st.planets[sh.b];
            Vec3 pp = planetPos(st, pl), sp{st.x, st.y, 0};
            float out = atan2f(pp.y - sp.y, pp.x - sp.x) + sh.ang0 + 0.04f * t;
            float dist = pl.size * 3.2f + pl.extent * 1.5f; // far enough to hold its moons and rings
            return {{pp.x + dist * cosf(out), pp.y + dist * sinf(out), dist * 0.32f}, lerp(pp, sp, 0.06f)};
        }
        case CHASE: {
            const Squad& sq = squads[sh.a];
            float hl = sqrtf(sq.hx * sq.hx + sq.hy * sq.hy) + 1e-6f, ux = sq.hx / hl, uy = sq.hy / hl;
            float side = 0.07f * sinf(sh.ang0 + 0.12f * t);
            return {{sq.cx - ux * 0.21f - uy * side, sq.cy - uy * 0.21f + ux * side, 0.105f}, {sq.cx + ux * 0.10f, sq.cy + uy * 0.10f, 0.04f}};
        }
        default: {
            float th = sh.ang0 + 0.03f * t;
            return {{1.5f * cosf(th), 1.5f * sinf(th), 0.78f}, {0, 0, -0.1f}};
        }
        }
    }

    void startCamera() {
        shot = Shot();
        shot.t0 = time; shot.ang0 = frand() * 2 * PI;
        prevShot = shot;
        shotCount = 0; ending = false;
        cam = poseOf(shot);
        camFwd = normalize(cam.tgt - cam.pos);
    }

    static constexpr ShotType ORDER[6] = {SYSTEM, PLANET, CHASE, SYSTEM, PLANET, WIDE};
    // A planet pass is a short hop within the system already in view, so it settles sooner.
    static float transOf(ShotType t) { return t == PLANET ? 4.0f : 6.0f; }

    void nextShot(float remaining, bool forDrop) {
        static const float DUR[] = {9, 8, 11, 10}; // by ShotType: WIDE, SYSTEM, PLANET, CHASE
        prevShot = shot;
        shot.t0 = time;
        shot.type = ORDER[shotCount++ % 6];
        // Not enough song left for another full shot: pull back and hold to the end.
        if (remaining < DUR[shot.type] + 8) { shot.type = WIDE; ending = true; }
        shot.trans = transOf(shot.type); shot.arc = shot.type == PLANET ? 0.05f : 0.14f;
        if (forDrop && !ending) {
            // Cross the whole constellation; the planet pass follows next.
            shot.type = SYSTEM; shotCount = 1;
            shot.trans = 4.6f; shot.arc = 0.24f;
        }
        shot.dur = DUR[shot.type];
        shot.ang0 = frand() * 2 * PI;
        if (shot.type == SYSTEM) {
            // Go where the music is: the loudest star we did not just visit. On a
            // drop, the farthest one instead, for the longest flight.
            int best = -1;
            for (int i = 0; i < NSTARS; i++) {
                if (i == lastStar) continue;
                auto score = [&](const Star& st) {
                    return forDrop ? hypotf(st.x - cam.pos.x, st.y - cam.pos.y) : st.glow;
                };
                if (best < 0 || score(stars[i]) > score(stars[best])) best = i;
            }
            shot.a = lastStar = best;
            // Arrive on the side we are coming from, so the approach does not cross the system.
            shot.ang0 = atan2f(cam.pos.y - stars[best].y, cam.pos.x - stars[best].x);
        } else if (shot.type == PLANET) {
            shot.a = lastStar; shot.b = 0;
            const auto& pls = stars[lastStar].planets;
            // The most interesting world (the one with the most travelling with it)
            // among those far enough out to visit without crowding the sun.
            shot.b = (int)pls.size() - 1;
            float clear = stars[lastStar].size * STAR_BERTH + 0.03f;
            for (size_t k = 0; k < pls.size(); k++)
                if (pls[k].r > clear && pls[k].extent > pls[shot.b].extent) shot.b = (int)k;
            shot.ang0 = (frand() < 0.5f ? -1 : 1) * (1.5f + 0.5f * frand());
        } else if (shot.type == CHASE) {
            shot.a = (int)(frand() * (NSQUADS - 0.01f));
        } else {
            shot.ang0 = atan2f(cam.pos.y, cam.pos.x);
        }
    }

    void stepCamera(float remaining, float toDrop) {
        // A new move may only begin once the last one has fully arrived, or the blend would jump.
        float age = time - shot.t0;
        bool settled = age > shot.trans + 1;
        if (!ending) {
            if (dropArmed && toDrop <= DROP_LEAD && settled) { nextShot(remaining, true); dropArmed = false; }
            // With a drop close ahead, hold this shot unless the next move can settle
            // in time, so the swoop is free to start on cue.
            else if (age > shot.dur && toDrop > transOf(ORDER[shotCount % 6]) + 1 + DROP_LEAD + 0.3f) nextShot(remaining, false);
        }
        if (toDrop > DROP_LEAD + 0.5f) dropArmed = true;
        float uu = clampf((time - shot.t0) / shot.trans, 0, 1);
        float e = uu * uu * uu * (uu * (uu * 6 - 15) + 10);
        Pose a = poseOf(prevShot), b = poseOf(shot);
        cam.pos = lerp(a.pos, b.pos, e);
        cam.tgt = lerp(a.tgt, b.tgt, e);
        // Rise a little mid-flight so long moves swoop over the plane rather than skim it.
        cam.pos.z += shot.arc * sinf(PI * e) * std::min(1.0f, length(b.pos - a.pos));
        // Never fly through a body.
        auto keepOut = [&](Vec3 c, float r) {
            Vec3 d = cam.pos - c;
            float l = length(d);
            if (l < r) cam.pos = c + normalize(d + Vec3{0, 0, 1e-4f}) * r;
        };
        for (const Star& st : stars) {
            keepOut({st.x, st.y, 0}, st.size * STAR_BERTH); // wide: a sun should never fill the frame
            for (const Planet& pl : st.planets) keepOut(planetPos(st, pl), pl.size * 2.0f);
        }
        cam.pos.z = std::max(cam.pos.z, 0.012f);
    }

    // The view eases toward the target and can only turn so fast. Without this, a
    // flight path that passes close over its own target would whip the view round.
    void turnCamera(float dt) {
        const float MAX_TURN = 0.9f; // radians per second
        Vec3 want = normalize(cam.tgt - cam.pos);
        Vec3 next = normalize(camFwd + (want - camFwd) * (1 - expf(-dt / 0.3f)));
        float ang = acosf(clampf(dot(next, camFwd), -1, 1)), cap = MAX_TURN * dt;
        if (ang > cap) next = normalize(camFwd + (next - camFwd) * (cap / ang));
        // Never look straight up or down: the horizon must stay level.
        next.z = clampf(next.z, -0.92f, 0.92f);
        camFwd = normalize(next);
    }

    void step(float dt, const Feat& f, const Palette& pal, float remaining, float toDrop) {
        time += dt;
        surge *= expf(-dt / 2.5f);
        while (!waves.empty() && time - waves.front().t0 > WAVE_LIFE) waves.erase(waves.begin());
        float lv[3] = {f.band[0], 0.5f * (f.band[1] + f.band[2]), std::max(f.band[3], f.band[4])};
        for (int k = 0; k < 3; k++) slv[k] += (lv[k] - slv[k]) * (1 - expf(-dt / 0.5f));
        moveStars(dt, f);
        stepFluid(dt, lv);
        stellarWind(dt, pal);
        feedPulse(dt, pal);
        stepShips(dt, lv, f.energy, pal);
        stepCamera(remaining, toDrop);
        turnCamera(dt);
    }
};

// ---------------------------------------------------------------- rendering

static const char* GLSL_HEAD = "#version 330 core\n";

// Camera shared by every shader: a position plus an orthonormal basis.
static const char* GLSL_CAMERA = R"(
uniform vec3 uCamPos, uRight, uUp, uFwd;
uniform float uTan, uAspect;
vec4 project(vec3 w) {
    vec3 v = w - uCamPos;
    float z = dot(v, uFwd);
    const float n = 0.002, f = 30.0;
    return vec4(dot(v, uRight) / (uTan * uAspect), dot(v, uUp) / uTan, (z * (f + n) - 2.0 * f * n) / (f - n), z);
}
)";

static const char* VS_FULL = R"(
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
})";

// Everything that is not a solid body: deep-sky stars, the nebula (the fluid's
// dye, as glowing sheets in the galactic plane), shockwaves, hyperlanes, the
// core and the stars' coronas.
static const char* FS_SKY = R"(
in vec2 vUV;
out vec4 frag;
uniform sampler2D uDye;
uniform float uTime, uPix;
uniform vec4 uLv; // bass, mid, high, energy
uniform vec3 uPulseCol, uTint;
uniform vec4 uWaves[8]; // origin xy, radius, amplitude
uniform vec4 uLanes[5]; // from xy, to xy
uniform vec4 uStars[6]; // x, y, radius, level
uniform vec3 uStarCols[6];

float hash1(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
vec2 hash2(vec2 p) { return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash1(i), hash1(i + vec2(1, 0)), f.x), mix(hash1(i + vec2(0, 1)), hash1(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + 17.0; a *= 0.5; }
    return s;
}
float fbm2(vec2 p) { return 0.6 * vnoise(p) + 0.4 * vnoise(p * 2.03 + 17.0); }
float stars(vec2 p, float scale, float seed) {
    vec2 g = p * scale, id = floor(g), f = fract(g) - 0.5;
    vec2 r = hash2(id + seed);
    float d = length(f - (r - 0.5) * 0.7);
    float b = pow(hash1(id * 1.7 + seed), 10.0);
    float tw = 0.65 + 0.35 * sin(uTime * (1.0 + r.x * 3.0) + r.y * 40.0);
    return b * tw * smoothstep(0.07, 0.0, d);
}
// Stars on the faces of a cube around the viewer, so they hold still as the camera turns.
float starLayer(vec3 d, float scale, float seed) {
    vec3 a = abs(d);
    vec2 uv; float face;
    if (a.x >= a.y && a.x >= a.z) { uv = d.yz / a.x; face = d.x > 0.0 ? 1.0 : 2.0; }
    else if (a.y >= a.z) { uv = d.xz / a.y; face = d.y > 0.0 ? 3.0 : 4.0; }
    else { uv = d.xy / a.z; face = d.z > 0.0 ? 5.0 : 6.0; }
    return stars(uv, scale, seed + face * 17.0);
}
void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    vec3 dir = normalize(uFwd + uRight * ndc.x * uTan * uAspect + uUp * ndc.y * uTan);
    vec3 col = vec3(0.0);

    // Deep sky; the highs make it glitter.
    float st = starLayer(dir, 30.0, 1.0) * 1.6 + starLayer(dir, 55.0, 2.0) + starLayer(dir, 90.0, 3.0) * 0.7;
    col += vec3(0.85, 0.9, 1.0) * st * 2.2 * (0.8 + 0.35 * uLv.z);
    col += uTint * 0.07 * fbm(dir.xy * 2.5 + dir.z * 1.7) * (0.5 + uLv.w);

    if (dir.z < -1e-3) {
        // Nebula: three stacked sheets of the same dye, for depth as the camera moves.
        for (int l = 0; l < 3; l++) {
            float fl = float(l), t = (-0.07 * fl - uCamPos.z) / dir.z;
            vec2 hit = uCamPos.xy + dir.xy * t;
            float fade = smoothstep(1.1, 0.35, length(hit)) * (0.2 + 0.8 * smoothstep(0.0, 0.45, -dir.z));
            if (fade <= 0.0) continue;
            vec2 q = hit * (1.0 - 0.06 * fl);
            vec2 warp = vec2(fbm2(q * 3.0 + fl * 5.0 + uTime * 0.03), fbm2(q * 3.0 + 9.0 - uTime * 0.03)) - 0.5;
            vec3 dye = texture(uDye, (q + warp * 0.04) * 0.5 + 0.5).rgb;
            float grain = 0.45 + 1.1 * fbm(q * 4.0 + warp * 2.0 + fl * 3.0);
            col += (1.0 - exp(-dye * 0.5)) * grain * fade * (0.36 - 0.09 * fl);
        }

        // Features drawn on the plane itself. px is one pixel's footprint there.
        float t0 = -uCamPos.z / dir.z;
        vec2 w = uCamPos.xy + dir.xy * t0;
        float d = length(w), px = t0 * uPix / max(-dir.z, 0.04);
        if (d < 3.0) {
            float ring = 0.0;
            for (int i = 0; i < 8; i++) {
                float wd = clamp(px * 1.2, 0.008, 0.03);
                float q = (length(w - uWaves[i].xy) - uWaves[i].z) / wd;
                ring += uWaves[i].w * exp(-q * q) * 0.008 / wd;
            }
            // Faded out near a low camera, where a passing ring would otherwise wash over
            // much of the frame several times a second.
            col += mix(uPulseCol, vec3(1.0), 0.4) * ring * 0.14 * smoothstep(0.08, 0.5, t0);

            // Hyperlanes: a tree linking every star, with pulses that ride the mids.
            float lane = 0.0;
            for (int i = 0; i < 5; i++) {
                vec2 a2 = uLanes[i].xy, ab = uLanes[i].zw - a2;
                float L = length(ab);
                float hh = clamp(dot(w - a2, ab) / (L * L), 0.0, 1.0);
                float dl = length(w - a2 - ab * hh);
                float lw = clamp(px * 1.5, 0.0005, 0.012), dim = min(1.0, 0.0022 / lw);
                float line = exp(-pow(dl / lw, 2.0)) * 0.3 * dim;
                float along = (fract(hh * L * 1.6 - uTime * 0.22 + float(i) * 0.37) - 0.5) / 0.05;
                float pulse = exp(-along * along) * exp(-pow(dl / (lw * 2.2), 2.0)) * (0.2 + 1.4 * uLv.y) * dim;
                lane += line + pulse;
            }
            col += mix(vec3(0.35, 0.75, 1.0), uPulseCol, 0.25) * lane * 0.55 * smoothstep(0.04, 0.3, t0);
        }
    }

    // Coronas and anamorphic flares; the star discs themselves are solid bodies.
    for (int i = 0; i < 6; i++) {
        vec3 v = vec3(uStars[i].xy, 0.0) - uCamPos;
        float tca = dot(v, dir);
        if (tca <= 0.0) continue;
        vec3 off = v - dir * tca;
        float s = uStars[i].z, lv = uStars[i].w, dd2 = dot(off, off);
        vec3 c = uStarCols[i];
        col += c * (0.11 + 0.28 * lv) * s * s / (dd2 + s * s * 0.6);
        col += c * lv * 0.08 * exp(-abs(dot(off, uUp)) / (s * 0.18)) * exp(-abs(dot(off, uRight)) / (s * (2.5 + 3.5 * lv)));
    }
    frag = vec4(col, 1.0);
})";

// Instanced quads, three vec4s each:
//   A = centre xyz, radius   B = rgb, param   C = kind, light direction xyz
// kind 0 planet/moon (param = PlanetType + seed)   kind 1 orbit trail in the plane
// (param = where the body is on it)   kind 2 star disc (param = level)   kind 3 ship
// kind 4 planetary rings in the plane (param = seed)
static const char* VS_SPRITE = R"(
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec4 aA;
layout(location = 2) in vec4 aB;
layout(location = 3) in vec4 aC;
out vec2 vC;
out vec3 vCol, vL;
flat out float vKind, vParam, vRadius;
void main() {
    float kind = aC.x, r = aA.w;
    float pad = kind < 0.5 ? 1.5 : (kind < 1.5 ? 1.1 : 1.0);
    // Ships stay points of light however close the camera gets.
    if (kind > 2.5) r = min(r, 0.011 * max(dot(aA.xyz - uCamPos, uFwd), 0.0) * uTan);
    vec3 w = (kind > 0.5 && kind < 1.5) || kind > 3.5 ? aA.xyz + vec3(aCorner * r * pad, 0.0)
                                      : aA.xyz + (uRight * aCorner.x + uUp * aCorner.y) * r * pad;
    gl_Position = project(w);
    if (kind < 0.5 || (kind > 1.5 && kind < 2.5)) {
        // Solid spheres take the depth of their near surface, so orbit lines pass behind them.
        vec4 nearer = project(w - uFwd * r);
        gl_Position.z = nearer.z / nearer.w * gl_Position.w;
    }
    vC = aCorner * pad; vCol = aB.rgb; vL = aC.yzw; vKind = kind; vParam = aB.w; vRadius = r;
})";

static const char* FS_SPRITE = R"(
in vec2 vC;
in vec3 vCol, vL;
flat in float vKind, vParam, vRadius;
out vec4 frag;
uniform float uGlowPass, uTime;

float hash3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float vnoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(i), hash3(i + vec3(1, 0, 0)), f.x), mix(hash3(i + vec3(0, 1, 0)), hash3(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash3(i + vec3(0, 0, 1)), hash3(i + vec3(1, 0, 1)), f.x), mix(hash3(i + vec3(0, 1, 1)), hash3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm3(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 4; i++) { s += a * vnoise3(p); p = p * 2.03 + 11.0; a *= 0.5; }
    return s;
}
void main() {
    float d = length(vC), fw = fwidth(d);
    if (vKind < 0.5) {
        float type = floor(vParam + 0.001), seed = fract(vParam);
        if (uGlowPass > 0.5) {
            // Atmosphere: a thin lit halo just outside the limb. Bare rock has none.
            if (d < 1.0) discard;
            float air = type < 0.5 ? 0.0 : (type < 1.5 ? 0.2 : (type > 2.5 && type < 3.5 ? 0.15 : 0.45));
            float lit = max(dot(normalize(uRight * vC.x + uUp * vC.y), vL) * 0.6 + 0.4, 0.0);
            frag = vec4(vCol * exp(-(d - 1.0) * 9.0) * lit * air, 0.0);
            return;
        }
        float cov = 1.0 - smoothstep(1.0 - fw, 1.0 + fw, d);
        if (cov < 0.02) discard;
        float mu = sqrt(max(0.0, 1.0 - d * d));
        // Normalised: past the antialiased limb d exceeds 1, and the glint's pow() would explode.
        vec3 n = normalize(uRight * vC.x + uUp * vC.y - uFwd * mu);
        // Surface coordinates turn with the planet, about the axis normal to its orbit.
        float spin = seed * 6.2831 + uTime * 0.04 * (0.5 + fract(seed * 7.3));
        vec3 p = vec3(cos(spin) * n.x - sin(spin) * n.y, sin(spin) * n.x + cos(spin) * n.y, n.z);
        float turb = fbm3(p * 3.0 + seed * 10.0);
        vec3 alb, emis = vec3(0.0);
        float gloss = 0.0;
        if (type < 0.5) {
            // Cratered rock (also the moons): mottled, pocked with darker floors.
            float crater = smoothstep(0.60, 0.70, vnoise3(p * 9.0 + seed * 17.0));
            alb = vCol * (0.55 + 0.7 * fbm3(p * 2.5 + seed * 31.0)) * (1.0 - 0.45 * crater);
        } else if (type < 1.5) {
            // Desert: dusty plains, darker basins, small polar caps.
            alb = mix(vCol * 0.5, vCol * 1.15 + 0.04, smoothstep(0.35, 0.65, fbm3(p * 4.0 + seed * 13.0)));
            alb = mix(alb, vec3(0.93), smoothstep(0.86, 0.94, abs(p.z) + (turb - 0.5) * 0.15));
        } else if (type < 2.5) {
            // Ocean world: seas that glint, continents, ice caps, drifting cloud.
            float land = smoothstep(0.50, 0.56, fbm3(p * 3.5 + seed * 20.0));
            vec3 ground = mix(vec3(0.15, 0.36, 0.14), vec3(0.56, 0.46, 0.28), fbm3(p * 7.0 + seed * 5.0));
            alb = mix(vCol, ground, land);
            alb = mix(alb, vec3(0.95), smoothstep(0.80, 0.90, abs(p.z) + (turb - 0.5) * 0.2));
            float cloud = smoothstep(0.52, 0.70, fbm3(p * 4.0 + vec3(uTime * 0.012, 0.0, seed * 40.0)));
            alb = mix(alb, vec3(1.0), cloud * 0.8);
            gloss = (1.0 - land) * (1.0 - cloud);
        } else if (type < 3.5) {
            // Lava: dark crust split by glowing fissures. The glow is steady.
            float ridge = 1.0 - abs(fbm3(p * 4.0 + seed * 9.0) * 2.0 - 1.0);
            alb = vCol * (0.6 + 0.8 * turb);
            emis = vec3(1.0, 0.33, 0.06) * smoothstep(0.80, 0.97, ridge) * 1.3;
        } else if (type < 4.5) {
            // Gas giant: stirred latitude bands, one off-colour belt, one great storm.
            float lat = p.z * (7.0 + 6.0 * fract(seed * 5.1)) + (turb - 0.5) * 3.5;
            alb = mix(vCol * 0.55, mix(vCol, vec3(1.0, 0.95, 0.85), 0.5), sin(lat) * 0.5 + 0.5);
            alb = mix(alb, vCol.brg * 0.8 + 0.1, 0.25 * smoothstep(0.6, 0.9, sin(lat * 0.37 + seed * 6.0)));
            vec3 eye = normalize(vec3(cos(seed * 40.0), sin(seed * 40.0), 0.35 - 0.7 * fract(seed * 9.0)));
            alb = mix(alb, vCol * vec3(1.2, 0.7, 0.6), smoothstep(0.30, 0.12, length((p - eye) * vec3(1.0, 1.0, 2.2))));
        } else {
            // Ice giant: smooth and pale, the faintest of bands.
            float b = sin(p.z * 5.0 + (turb - 0.5) * 1.5) * 0.5 + 0.5;
            alb = mix(vCol * 0.75, mix(vCol, vec3(1.0), 0.35), b * 0.6 + 0.2 * turb);
        }
        float nl = dot(n, vL);
        vec3 c = alb * (0.06 + 0.08 * mu + 1.35 * pow(clamp((nl + 0.3) / 1.3, 0.0, 1.0), 1.6)) + emis;
        c += vec3(1.0, 0.97, 0.9) * pow(clamp(dot(n, normalize(vL - uFwd + 1e-4)), 0.0, 1.0), 50.0) * 0.4 * gloss * step(0.0, nl);
        if (type > 0.5) c += vCol * pow(1.0 - mu, 3.0) * smoothstep(-0.2, 0.5, nl) * 0.5;
        frag = vec4(c * cov, cov);
    } else if (vKind < 1.5) {
        // Long-exposure trail: bright just behind the body, fading round the orbit.
        // Widened to stay a pixel across, but dimmed to match and dropped once it is
        // too foreshortened to resolve (otherwise it floods the whole quad edge-on).
        // A line about two pixels wide at any distance: never a fat band up close,
        // and never wider than the quad has room for.
        float foot = fw * vRadius * 0.8, wd = min(clamp(foot * 1.6, 0.0003, 0.006), 0.03 * vRadius);
        float q = (d - 1.0) * vRadius / wd;
        float behind = mod(vParam - atan(vC.y, vC.x), 6.2831853) / 6.2831853;
        float keep = min(1.0, 0.0016 / wd) * smoothstep(0.03, 0.006, foot);
        // The trail starts a little behind the body (vL.x = its radius), not through it.
        float arc = min(behind, 1.0 - behind) * 6.2831853 * vRadius;
        keep *= smoothstep(vL.x * 1.2, vL.x * 3.0, arc);
        frag = vec4(vCol * exp(-q * q) * (pow(1.0 - behind, 3.0) * 0.8 + 0.06) * keep, 0.0);
    } else if (vKind < 2.5) {
        float cov = 1.0 - smoothstep(1.0 - fw, 1.0 + fw, d);
        if (cov < 0.02) discard;
        float mu = sqrt(max(0.0, 1.0 - d * d));
        vec3 n = normalize(uRight * vC.x + uUp * vC.y - uFwd * mu);
        float gran = fbm3(n * 7.0 + uTime * 0.08);
        vec3 c = mix(vCol, vec3(1.0), 0.65 * mu) * (0.45 + 0.55 * mu) * (0.75 + 0.5 * gran) * (1.0 + 0.5 * vParam);
        frag = vec4(c * cov, cov);
    } else if (vKind < 3.5) {
        frag = vec4(vCol * (exp(-d * d * 5.0) + 2.0 * exp(-d * d * 40.0)), 0.0);
    } else {
        // Rings: concentric bands with gaps, from 0.55 to 1.0 of the quad.
        float band = smoothstep(0.55, 0.58, d) * (1.0 - smoothstep(0.97, 1.0, d));
        float g = floor(d * 46.0);
        float dens = 0.35 + 0.65 * fract(sin(g * 12.9898 + vParam * 78.233) * 43758.5453);
        dens *= 1.0 - smoothstep(0.02, 0.0, abs(d - 0.80)) * 0.9; // one clear division
        frag = vec4(vCol * band * dens * 0.55, 0.0);
    }
})";

// Ship trails: one quad per segment, A = start xyz, alpha   B = end xyz, alpha
// C = rgb, width in world units. Widened in screen space so far ribbons stay visible.
static const char* VS_TRAIL = R"(
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec4 aA;
layout(location = 2) in vec4 aB;
layout(location = 3) in vec4 aC;
uniform float uResY;
out vec3 vCol;
out float vSide;
void main() {
    vec4 ca = project(aA.xyz), cb = project(aB.xyz);
    if (ca.w < 0.004 || cb.w < 0.004) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vCol = vec3(0.0); vSide = 0.0; return; }
    float t = aCorner.x * 0.5 + 0.5;
    vec4 c = mix(ca, cb, t);
    vec2 dir = (cb.xy / cb.w - ca.xy / ca.w) * vec2(uAspect, 1.0);
    dir = dir / max(length(dir), 1e-6);
    float px = clamp(aC.w / (c.w * uTan) * uResY * 0.5, 1.1 * uResY / 1080.0, 3.2 * uResY / 1080.0);
    float wN = px * 2.0 / uResY;
    c.xy += vec2(-dir.y, dir.x) / vec2(uAspect, 1.0) * wN * aCorner.y * c.w;
    gl_Position = c;
    vCol = aC.rgb * mix(aA.w, aB.w, t);
    vSide = aCorner.y;
})";

static const char* FS_TRAIL = R"(
in vec3 vCol;
in float vSide;
out vec4 frag;
void main() { frag = vec4(vCol * exp(-vSide * vSide * 3.0), 0.0); })";

static const char* FS_BLUR = R"(
in vec2 vUV;
out vec4 frag;
uniform sampler2D uSrc;
uniform vec2 uDir;
uniform float uKnee;
vec3 tap(vec2 uv) {
    // Ceiling (and NaN guard): one runaway pixel must never bloom into a flash.
    vec3 c = clamp(texture(uSrc, uv).rgb, 0.0, 6.0);
    if (uKnee > 0.0) c *= smoothstep(uKnee * 0.5, uKnee * 2.0, dot(c, vec3(0.2126, 0.7152, 0.0722)));
    return c;
}
void main() {
    vec3 c = tap(vUV) * 0.227027;
    c += (tap(vUV + uDir * 1.384615) + tap(vUV - uDir * 1.384615)) * 0.316216;
    c += (tap(vUV + uDir * 3.230769) + tap(vUV - uDir * 3.230769)) * 0.070270;
    frag = vec4(c, 1.0);
})";

static const char* FS_COMPOSITE = R"(
in vec2 vUV;
out vec4 frag;
uniform sampler2D uScene, uB1, uB2, uB3, uB4;
uniform float uFade, uTime;
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
vec3 bloom(vec2 uv) {
    return texture(uB1, uv).rgb * 0.5 + texture(uB2, uv).rgb * 0.6 + texture(uB3, uv).rgb * 0.7 + texture(uB4, uv).rgb * 0.8;
}
vec3 sampleAll(vec2 uv) { return clamp(texture(uScene, uv).rgb, 0.0, 6.0) + bloom(uv) * 0.13; }
void main() {
    vec2 c = vUV - 0.5;
    // A touch of chromatic split toward the corners.
    float ca = 0.0010 * dot(c, c) * 4.0;
    vec3 col = vec3(sampleAll(vUV + c * ca).r, sampleAll(vUV).g, sampleAll(vUV - c * ca).b);
    col = aces(col * 0.95);
    col *= 1.0 - 0.55 * dot(c, c) * 1.6;
    col = pow(col, vec3(1.0 / 2.2));
    col += (fract(sin(dot(vUV * 917.0 + uTime, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0 * 2.0;
    frag = vec4(col * uFade, 1.0);
})";

static GLuint compile(GLenum type, const std::string& src) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(s, sizeof log, nullptr, log); fprintf(stderr, "shader error:\n%s\n", log); exit(1); }
    return s;
}
static GLuint program(const char* vs, const char* fs) {
    GLuint p = glCreateProgram();
    std::string head = std::string(GLSL_HEAD) + GLSL_CAMERA;
    glAttachShader(p, compile(GL_VERTEX_SHADER, head + vs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, head + fs));
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, nullptr, log); fprintf(stderr, "link error:\n%s\n", log); exit(1); }
    return p;
}
static GLint U(GLuint p, const char* name) { return glGetUniformLocation(p, name); }

struct Target { GLuint fbo = 0, tex = 0; int w = 0, h = 0; };
static Target makeTarget(int w, int h, GLint internal, GLenum type, bool depth = false) {
    Target t; t.w = w; t.h = h;
    glGenTextures(1, &t.tex);
    glBindTexture(GL_TEXTURE_2D, t.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, GL_RGBA, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &t.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
    if (depth) {
        GLuint rb;
        glGenRenderbuffers(1, &rb);
        glBindRenderbuffer(GL_RENDERBUFFER, rb);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);
    }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "framebuffer incomplete\n"); exit(1); }
    return t;
}

struct Renderer {
    int W = 0, H = 0;
    GLuint pSky, pSprite, pTrail, pBlur, pComp;
    GLuint vaoFull, vaoInst, vboInst, dyeTex;
    Target scene, out, blurTmp[4], blurOut[4];
    std::vector<float> solid, glow, trails, dyeBuf;

    void init(int w, int h, const Sim& sim) {
        W = w; H = h;
        pSky = program(VS_FULL, FS_SKY);
        pSprite = program(VS_SPRITE, FS_SPRITE);
        pTrail = program(VS_TRAIL, FS_TRAIL);
        pBlur = program(VS_FULL, FS_BLUR);
        pComp = program(VS_FULL, FS_COMPOSITE);
        glGenVertexArrays(1, &vaoFull);

        static const float quad[] = {-1, -1, 1, -1, -1, 1, 1, 1};
        GLuint vboQuad;
        glGenVertexArrays(1, &vaoInst);
        glBindVertexArray(vaoInst);
        glGenBuffers(1, &vboQuad);
        glBindBuffer(GL_ARRAY_BUFFER, vboQuad);
        glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glGenBuffers(1, &vboInst);
        glBindBuffer(GL_ARRAY_BUFFER, vboInst);
        for (int a = 1; a <= 3; a++) {
            glEnableVertexAttribArray(a);
            glVertexAttribPointer(a, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)((a - 1) * 4 * sizeof(float)));
            glVertexAttribDivisor(a, 1);
        }
        glBindVertexArray(0);

        glGenTextures(1, &dyeTex);
        glBindTexture(GL_TEXTURE_2D, dyeTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, sim.nx, sim.ny, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        scene = makeTarget(W, H, GL_RGBA16F, GL_HALF_FLOAT, true);
        out = makeTarget(W, H, GL_RGBA8, GL_UNSIGNED_BYTE);
        for (int i = 0; i < 4; i++) {
            int bw = std::max(1, W >> (i + 1)), bh = std::max(1, H >> (i + 1));
            blurTmp[i] = makeTarget(bw, bh, GL_RGBA16F, GL_HALF_FLOAT);
            blurOut[i] = makeTarget(bw, bh, GL_RGBA16F, GL_HALF_FLOAT);
        }
    }

    static void push(std::vector<float>& v, Vec3 c, float r, const float rgb[3], float gain, float param, float kind, Vec3 l = {}) {
        float f[12] = {c.x, c.y, c.z, r, rgb[0] * gain, rgb[1] * gain, rgb[2] * gain, param, kind, l.x, l.y, l.z};
        v.insert(v.end(), f, f + 12);
    }

    // Renders one frame into `out`.
    void draw(const Sim& sim, const Feat& f, const Palette& pal, float fade) {
        const float* lv = sim.slv;
        float aspect = (float)W / H;
        Vec3 fwd = sim.camFwd, right = normalize(cross(fwd, {0, 0, 1})), up = cross(right, fwd);
        // Tall frames get a wider lens so the scene still fits across.
        float tanH = 0.466f / sqrtf(std::min(1.0f, aspect));
        auto camera = [&](GLuint prog) {
            glUseProgram(prog);
            glUniform3f(U(prog, "uCamPos"), sim.cam.pos.x, sim.cam.pos.y, sim.cam.pos.z);
            glUniform3f(U(prog, "uRight"), right.x, right.y, right.z);
            glUniform3f(U(prog, "uUp"), up.x, up.y, up.z);
            glUniform3f(U(prog, "uFwd"), fwd.x, fwd.y, fwd.z);
            glUniform1f(U(prog, "uTan"), tanH);
            glUniform1f(U(prog, "uAspect"), aspect);
        };
        auto drawInstances = [&](const std::vector<float>& inst) {
            if (inst.empty()) return;
            glBindVertexArray(vaoInst);
            glBindBuffer(GL_ARRAY_BUFFER, vboInst);
            glBufferData(GL_ARRAY_BUFFER, inst.size() * sizeof(float), inst.data(), GL_STREAM_DRAW);
            glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)(inst.size() / 12));
        };

        dyeBuf.resize((size_t)sim.nx * sim.ny * 3);
        for (size_t c = 0; c < (size_t)sim.nx * sim.ny; c++)
            for (int k = 0; k < 3; k++) dyeBuf[c * 3 + k] = sim.dye[k][c];
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, dyeTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sim.nx, sim.ny, GL_RGB, GL_FLOAT, dyeBuf.data());

        // ---- sky, nebula and everything painted on the plane
        glBindFramebuffer(GL_FRAMEBUFFER, scene.fbo);
        glViewport(0, 0, W, H);
        glDepthMask(GL_TRUE);
        glClear(GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        camera(pSky);
        glUniform1i(U(pSky, "uDye"), 0);
        glUniform1f(U(pSky, "uTime"), sim.time);
        glUniform1f(U(pSky, "uPix"), 2 * tanH / H);
        glUniform4f(U(pSky, "uLv"), lv[0], lv[1], lv[2], f.energy);
        glUniform3fv(U(pSky, "uPulseCol"), 1, pal.star);
        glUniform3fv(U(pSky, "uTint"), 1, pal.tint);
        float wv[MAXWAVES * 4] = {0};
        for (size_t i = 0; i < sim.waves.size(); i++) {
            const Wave& w = sim.waves[i];
            float age = sim.time - w.t0;
            wv[i * 4] = w.x; wv[i * 4 + 1] = w.y;
            wv[i * 4 + 2] = 0.05f + WAVE_SPEED * age;
            wv[i * 4 + 3] = w.strength * w.strength * smooth01(age / 0.15f) * expf(-3.0f * age);
        }
        glUniform4fv(U(pSky, "uWaves"), MAXWAVES, wv);
        float lv4[(NSTARS - 1) * 4];
        for (int e = 0; e < NSTARS - 1; e++) {
            const Star &a = sim.stars[sim.lanes[e][0]], &b = sim.stars[sim.lanes[e][1]];
            lv4[e * 4] = a.x; lv4[e * 4 + 1] = a.y; lv4[e * 4 + 2] = b.x; lv4[e * 4 + 3] = b.y;
        }
        glUniform4fv(U(pSky, "uLanes"), NSTARS - 1, lv4);
        float starv[NSTARS * 4], starc[NSTARS * 3];
        for (int i = 0; i < NSTARS; i++) {
            const Star& st = sim.stars[i];
            starv[i * 4] = st.x; starv[i * 4 + 1] = st.y; starv[i * 4 + 2] = st.size; starv[i * 4 + 3] = st.glow;
            starColor(st, pal, &starc[i * 3]);
        }
        glUniform4fv(U(pSky, "uStars"), NSTARS, starv);
        glUniform3fv(U(pSky, "uStarCols"), NSTARS, starc);
        glBindVertexArray(vaoFull);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // ---- gather bodies
        solid.clear(); glow.clear(); trails.clear();
        static const float GREY[3] = {0.62f, 0.61f, 0.60f};
        for (int i = 0; i < NSTARS; i++) {
            const Star& st = sim.stars[i];
            Vec3 sp{st.x, st.y, 0};
            push(solid, sp, st.size, &starc[i * 3], 1, st.glow, 2);
            float lit = 0.8f + 0.6f * st.glow, trail = 0.4f + 0.7f * st.glow;
            for (const Planet& pl : st.planets) {
                float c[3];
                planetColor(st, pl, pal, c);
                // Lava crust is nearly black; give its trail and rings something to show.
                float bright[3] = {c[0] * 0.7f + 0.25f, c[1] * 0.7f + 0.25f, c[2] * 0.7f + 0.25f};
                Vec3 pp = planetPos(st, pl), l = normalize(sp - pp);
                push(solid, pp, pl.size, c, lit, (float)pl.type + pl.seed, 0, l);
                push(glow, pp, pl.size, bright, lit, (float)pl.type + pl.seed, 0, l);
                push(glow, sp, pl.r, bright, trail, pl.phase, 1, {pl.size, 0, 0});
                if (pl.ringed) push(glow, pp, pl.size * 2.3f, bright, lit, pl.seed, 4);
                for (const Moon& mo : pl.moons) {
                    Vec3 mp{pp.x + mo.r * cosf(mo.phase), pp.y + mo.r * sinf(mo.phase), 0};
                    push(solid, mp, mo.size, GREY, lit, (float)ROCK + mo.seed, 0, normalize(sp - mp));
                    push(glow, pp, mo.r, GREY, trail * 0.45f, mo.phase, 1, {mo.size, 0, 0});
                }
            }
        }
        static const float SHIP_SIZE[3] = {0.0065f, 0.005f, 0.004f};
        static const float SHIP_WIDTH[3] = {0.0020f, 0.0015f, 0.0011f};
        float active = 0.3f + 0.7f * f.energy;
        for (const Ship& s : sim.ships) {
            float br = Sim::fadeOf(s, active) * (0.6f + 1.0f * lv[s.s]);
            if (br < 0.004f) continue;
            const float* c = pal.species[s.s];
            float hot[3] = {c[0] * 0.6f + 0.4f, c[1] * 0.6f + 0.4f, c[2] * 0.6f + 0.4f};
            push(glow, {s.x, s.y, s.z}, SHIP_SIZE[s.s], hot, br * 2.2f, 0, 3);
            Vec3 prev{s.x, s.y, s.z};
            float aPrev = 1;
            for (int k = 0; k < TRAIL; k++) {
                Vec3 cur{s.trail[k][0], s.trail[k][1], s.trail[k][2]};
                float aCur = powf(1 - (k + 1.0f) / TRAIL, 1.6f);
                float seg[12] = {prev.x, prev.y, prev.z, aPrev, cur.x, cur.y, cur.z, aCur,
                                 c[0] * br * 0.55f, c[1] * br * 0.55f, c[2] * br * 0.55f, SHIP_WIDTH[s.s]};
                trails.insert(trails.end(), seg, seg + 12);
                prev = cur; aPrev = aCur;
            }
        }

        // ---- solid bodies write depth; everything luminous is then tested against it
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        camera(pSprite);
        glUniform1f(U(pSprite, "uTime"), sim.time);
        glUniform1f(U(pSprite, "uGlowPass"), 0);
        drawInstances(solid);
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_ONE, GL_ONE);
        glUniform1f(U(pSprite, "uGlowPass"), 1);
        drawInstances(glow);
        camera(pTrail);
        glUniform1f(U(pTrail, "uResY"), (float)H);
        drawInstances(trails);
        glDepthMask(GL_TRUE);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);

        // ---- bloom pyramid
        glUseProgram(pBlur);
        glUniform1i(U(pBlur, "uSrc"), 0);
        glBindVertexArray(vaoFull);
        GLuint src = scene.tex;
        for (int i = 0; i < 4; i++) {
            glBindFramebuffer(GL_FRAMEBUFFER, blurTmp[i].fbo);
            glViewport(0, 0, blurTmp[i].w, blurTmp[i].h);
            glBindTexture(GL_TEXTURE_2D, src);
            glUniform2f(U(pBlur, "uDir"), 1.0f / blurTmp[i].w, 0);
            glUniform1f(U(pBlur, "uKnee"), i == 0 ? 0.8f : 0.0f);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindFramebuffer(GL_FRAMEBUFFER, blurOut[i].fbo);
            glBindTexture(GL_TEXTURE_2D, blurTmp[i].tex);
            glUniform2f(U(pBlur, "uDir"), 0, 1.0f / blurTmp[i].h);
            glUniform1f(U(pBlur, "uKnee"), 0.0f);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            src = blurOut[i].tex;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
        glViewport(0, 0, W, H);
        glUseProgram(pComp);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, scene.tex);
        glUniform1i(U(pComp, "uScene"), 0);
        static const char* names[4] = {"uB1", "uB2", "uB3", "uB4"};
        for (int i = 0; i < 4; i++) {
            glActiveTexture(GL_TEXTURE1 + i); glBindTexture(GL_TEXTURE_2D, blurOut[i].tex);
            glUniform1i(U(pComp, names[i]), 1 + i);
        }
        glActiveTexture(GL_TEXTURE0);
        glUniform1f(U(pComp, "uFade"), fade);
        glUniform1f(U(pComp, "uTime"), sim.time);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
};

// ---------------------------------------------------------------- playback

struct Player {
    const std::vector<float>* pcm = nullptr;
    std::atomic<long long> pos{0}; // in stereo frames
    std::atomic<bool> paused{false};
};
static void audioCallback(void* ud, Uint8* stream, int len) {
    Player* pl = (Player*)ud;
    float* outp = (float*)stream;
    int frames = len / (int)(2 * sizeof(float));
    long long pos = pl->pos.load(), total = pl->pcm ? (long long)pl->pcm->size() / 2 : 0;
    for (int i = 0; i < frames; i++) {
        if (pl->paused || pos >= total) { outp[2 * i] = outp[2 * i + 1] = 0; continue; }
        outp[2 * i] = (*pl->pcm)[2 * pos]; outp[2 * i + 1] = (*pl->pcm)[2 * pos + 1];
        pos++;
    }
    pl->pos.store(pos);
}

static float fadeAt(double t, double dur) { return smooth01((float)(t / 1.2)) * smooth01((float)((dur - t) / 2.0)); }

// Fires any kicks in (from, to].
static void fireKicks(Sim& sim, const Track& tr, size_t& next, double to) {
    while (next < tr.kicks.size() && tr.kicks[next].t <= to) sim.kick(tr.kicks[next++].strength);
}
// Fires any drops in (from, to] and returns how long until the next one.
static float fireDrops(Sim& sim, const Track& tr, size_t& next, double to) {
    while (next < tr.drops.size() && tr.drops[next] <= to) { sim.drop(); next++; }
    return next < tr.drops.size() ? (float)(tr.drops[next] - to) : 1e9f;
}
static size_t dropIndexAt(const Track& tr, double t) {
    size_t i = 0;
    while (i < tr.drops.size() && tr.drops[i] <= t) i++;
    return i;
}
static size_t kickIndexAt(const Track& tr, double t) {
    size_t i = 0;
    while (i < tr.kicks.size() && tr.kicks[i].t <= t) i++;
    return i;
}

static void usage() {
    printf("usage: viz [options] song.mp3 [more songs...]\n"
           "  --render out.mp4     render offline to a video with the song muxed in (single song)\n"
           "  --size WxH           resolution (default 1280x720 window, 1920x1080 render; try 1080x1920)\n"
           "  --fps N              render frame rate (default 60)\n"
           "  --start S            start time in seconds\n"
           "  --duration S         render only this many seconds\n"
           "  --pulse SOURCE       what the centre star pulses to: synth, melody, kick, drums, hats, bass,\n"
           "                       other, vocals (stems from `make stems`), mix, an audio file, or several\n"
           "                       joined with + (synth+kick). Default: synth if the song has stems, else mix\n"
           "  --pulses             list every pulse source for the given songs with its hit count and\n"
           "                       steadiness, then exit\n"
           "  --palette N          force palette 0..%d (default: picked from the song name)\n"
           "  --ships N            number of ships (default 96)\n"
           "  --fullscreen\n"
           "keys: space pause, left/right seek 5s, n next song, c cycle palette, f fullscreen, q/esc quit\n",
           NPALETTES - 1);
}

int main(int argc, char** argv) {
    std::vector<std::string> files;
    std::string renderPath, pulse;
    int W = 0, H = 0, fps = 60, paletteOverride = -1, nships = 96;
    double start = 0, duration = -1;
    bool fullscreen = false, listPulses = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { usage(); exit(1); } return argv[++i]; };
        if (a == "--render") renderPath = next();
        else if (a == "--size") { if (sscanf(next(), "%dx%d", &W, &H) != 2) { usage(); return 1; } }
        else if (a == "--fps") fps = atoi(next());
        else if (a == "--start") start = atof(next());
        else if (a == "--duration") duration = atof(next());
        else if (a == "--palette") paletteOverride = atoi(next());
        else if (a == "--pulse") pulse = next();
        else if (a == "--pulses") listPulses = true;
        else if (a == "--ships") nships = atoi(next());
        else if (a == "--fullscreen") fullscreen = true;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else files.push_back(a);
    }
    if (files.empty()) { usage(); return 1; }
    if (listPulses) {
        for (const std::string& f : files) {
            Track t;
            if (loadTrack(f, t, paletteOverride, pulse)) comparePulses(t);
        }
        return 0;
    }
    bool rendering = !renderPath.empty();
    if (W <= 0 || H <= 0) { W = rendering ? 1920 : 1280; H = rendering ? 1080 : 720; }
    if (rendering) { W &= ~1; H &= ~1; } // yuv420 needs even dimensions
    fps = std::max(1, fps);

    size_t trackIdx = 0;
    Track track;
    if (!loadTrack(files[0], track, paletteOverride, pulse)) return 1;

    if (SDL_Init(SDL_INIT_VIDEO | (rendering ? 0 : SDL_INIT_AUDIO)) != 0) { fprintf(stderr, "SDL: %s\n", SDL_GetError()); return 1; }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_ALLOW_HIGHDPI | (rendering ? SDL_WINDOW_HIDDEN : SDL_WINDOW_RESIZABLE);
    if (fullscreen && !rendering) flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    SDL_Window* win = SDL_CreateWindow("muzak", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, rendering ? 64 : W,
                                       rendering ? 64 : H, flags);
    if (!win) { fprintf(stderr, "window: %s\n", SDL_GetError()); return 1; }
    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) { fprintf(stderr, "GL context: %s\n", SDL_GetError()); return 1; }

    // Live: draw at the display's real pixel density.
    int RW = W, RH = H;
    if (!rendering) {
        int dw, dh, ww, wh;
        SDL_GL_GetDrawableSize(win, &dw, &dh);
        SDL_GetWindowSize(win, &ww, &wh);
        float scale = ww > 0 ? (float)dw / ww : 1;
        RW = (int)(W * scale); RH = (int)(H * scale);
    }

    Sim sim;
    if (H > W) sim.lift = 2.2f;
    rngState = track.seed;
    sim.init(nships);
    Renderer ren;
    ren.init(RW, RH, sim);

    // Let the squadrons draw out their trails before the first frame.
    Feat quiet = track.at(start);
    for (int i = 0; i < 180; i++) sim.step(1.0f / 60, quiet, PALETTES[track.palette], 1e9f, 1e9f);
    sim.startCamera();

    if (rendering) {
        if (duration < 0 || start + duration > track.dur) duration = track.dur - start;
        if (duration <= 0) { fprintf(stderr, "nothing to render\n"); return 1; }
        char cmd[4096];
        snprintf(cmd, sizeof cmd,
                 "ffmpeg -y -v error -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - -ss %.3f -t %.3f -i %s "
                 "-map 0:v -map 1:a -vf vflip -c:v libx264 -preset medium -crf 16 -pix_fmt yuv420p "
                 "-c:a aac -b:a 256k -shortest -movflags +faststart %s",
                 RW, RH, fps, start, duration, shq(track.path).c_str(), shq(renderPath).c_str());
        FILE* enc = popen(cmd, "w");
        if (!enc) { fprintf(stderr, "could not start ffmpeg\n"); return 1; }
        std::vector<unsigned char> px((size_t)RW * RH * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        long long nframes = (long long)(duration * fps);
        int sub = fps < 50 ? 2 : 1;
        size_t nextKick = kickIndexAt(track, start), nextDrop = dropIndexAt(track, start);
        for (long long fr = 0; fr < nframes; fr++) {
            for (int s = 1; s <= sub; s++) {
                double t = start + (fr + (double)s / sub) / fps;
                fireKicks(sim, track, nextKick, t);
                float toDrop = fireDrops(sim, track, nextDrop, t);
                sim.step(1.0f / (fps * sub), track.at(t), PALETTES[track.palette], (float)(start + duration - t), toDrop);
            }
            double t = start + (double)(fr + 1) / fps;
            ren.draw(sim, track.at(t), PALETTES[track.palette], fadeAt(t - start, duration));
            glBindFramebuffer(GL_FRAMEBUFFER, ren.out.fbo);
            glReadPixels(0, 0, RW, RH, GL_RGB, GL_UNSIGNED_BYTE, px.data());
            if (fwrite(px.data(), 1, px.size(), enc) != px.size()) { fprintf(stderr, "\nffmpeg stopped accepting frames\n"); break; }
            if (fr % fps == 0) { printf("\rrendering %s  %3d%%", renderPath.c_str(), (int)(100 * fr / nframes)); fflush(stdout); }
        }
        int rc = pclose(enc);
        printf("\r%s %s                    \n", rc == 0 ? "wrote" : "ffmpeg failed writing", renderPath.c_str());
        SDL_Quit();
        return rc == 0 ? 0 : 1;
    }

    // ---- live
    SDL_GL_SetSwapInterval(1);
    Player player;
    player.pcm = &track.pcm;
    player.pos = (long long)(start * SR);
    SDL_AudioSpec want{}, have{};
    want.freq = SR; want.format = AUDIO_F32SYS; want.channels = 2; want.samples = 1024;
    want.callback = audioCallback; want.userdata = &player;
    SDL_AudioDeviceID dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
    if (!dev) { fprintf(stderr, "audio: %s\n", SDL_GetError()); return 1; }
    double latency = (double)have.samples / SR;
    SDL_PauseAudioDevice(dev, 0);
    SDL_SetWindowTitle(win, ("muzak: " + track.name).c_str());

    double t = start; // smoothed song clock
    size_t nextKick = kickIndexAt(track, t), nextDrop = dropIndexAt(track, t);
    Uint64 prev = SDL_GetPerformanceCounter();
    bool running = true;
    auto seekTo = [&](double to) {
        to = clampf((float)to, 0, (float)track.dur);
        player.pos = (long long)(to * SR);
        t = to; nextKick = kickIndexAt(track, t); nextDrop = dropIndexAt(track, t);
    };
    auto openTrack = [&](size_t idx) {
        SDL_LockAudioDevice(dev);
        player.pcm = nullptr;
        SDL_UnlockAudioDevice(dev);
        if (!loadTrack(files[idx], track, paletteOverride, pulse)) return false;
        SDL_LockAudioDevice(dev);
        player.pcm = &track.pcm; player.pos = 0;
        SDL_UnlockAudioDevice(dev);
        trackIdx = idx; t = 0; nextKick = 0; nextDrop = 0;
        prev = SDL_GetPerformanceCounter();
        SDL_SetWindowTitle(win, ("muzak: " + track.name).c_str());
        return true;
    };

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type != SDL_KEYDOWN) continue;
            switch (e.key.keysym.sym) {
            case SDLK_ESCAPE: case SDLK_q: running = false; break;
            case SDLK_SPACE: player.paused = !player.paused; break;
            case SDLK_LEFT: seekTo(t - 5); break;
            case SDLK_RIGHT: seekTo(t + 5); break;
            case SDLK_c: track.palette = (track.palette + 1) % NPALETTES; break;
            case SDLK_f:
                fullscreen = !fullscreen;
                SDL_SetWindowFullscreen(win, fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
                break;
            case SDLK_n: if (!openTrack((trackIdx + 1) % files.size())) running = false; break;
            }
        }
        Uint64 now = SDL_GetPerformanceCounter();
        float dt = clampf((float)((double)(now - prev) / SDL_GetPerformanceFrequency()), 1.0f / 240, 1.0f / 30);
        prev = now;

        if (!player.paused) {
            // Free-run between audio callbacks, nudging toward the device clock.
            double audioT = (double)player.pos.load() / SR - latency;
            t += dt;
            double err = audioT - t;
            if (fabs(err) > 0.15) t = audioT; else t += err * 0.06;
            if (t >= track.dur) {
                if (trackIdx + 1 >= files.size() || !openTrack(trackIdx + 1)) break;
                continue;
            }
            fireKicks(sim, track, nextKick, t);
            float toDrop = fireDrops(sim, track, nextDrop, t);
            sim.step(dt, track.at(t), PALETTES[track.palette], (float)(track.dur - t), toDrop);
        }
        ren.draw(sim, track.at(t), PALETTES[track.palette], fadeAt(t, track.dur));

        // Letterbox the fixed-size frame into whatever the window is now.
        int dw, dh;
        SDL_GL_GetDrawableSize(win, &dw, &dh);
        float s = std::min((float)dw / RW, (float)dh / RH);
        int bw = (int)(RW * s), bh = (int)(RH * s), x0 = (dw - bw) / 2, y0 = (dh - bh) / 2;
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glViewport(0, 0, dw, dh);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, ren.out.fbo);
        glBlitFramebuffer(0, 0, RW, RH, x0, y0, x0 + bw, y0 + bh, GL_COLOR_BUFFER_BIT, GL_LINEAR);
        SDL_GL_SwapWindow(win);
    }
    SDL_CloseAudioDevice(dev);
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
