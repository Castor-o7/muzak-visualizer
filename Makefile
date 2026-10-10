CXX ?= clang++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra $(shell sdl2-config --cflags)
LDLIBS = $(shell sdl2-config --libs) -framework OpenGL

SONG ?= ../funky-er.mp3
# What the centre star pulses to: synth (default when the song has stems),
# melody, kick, drums, hats, bass, other, vocals, mix, a path to an audio file,
# or several joined with + (PULSE=synth+kick).
PULSE ?=
PULSEFLAG = $(if $(PULSE),--pulse $(PULSE))

viz: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

# make run SONG=../dontfitright.mp3
run: viz
	./viz $(PULSEFLAG) "$(SONG)"

# make video SONG=../dontfitright.mp3   -> dontfitright.mp4 (1080p60)
video: viz
	./viz $(PULSEFLAG) --render "$(basename $(notdir $(SONG))).mp4" "$(SONG)"

# 1080x1920 for Reels / Shorts / TikTok
vertical: viz
	./viz $(PULSEFLAG) --render "$(basename $(notdir $(SONG)))-vertical.mp4" --size 1080x1920 "$(SONG)"

# Separate a song into stems (drums, bass, other, vocals) next to it, in
# stems/htdemucs/<song>/. Needs uv; the first run downloads Demucs and its model.
stems:
	cd "$(dir $(SONG))" && uvx --python 3.11 --with "torch==2.5.1" --with "torchaudio==2.5.1" --with soundfile \
		demucs -d cpu -n htdemucs -o stems "$(notdir $(SONG))"

# Compare the pulse sources for a song: hits per minute and how steady they are.
pulses: viz
	./viz --pulses "$(SONG)"

clean:
	rm -f viz

.PHONY: run video vertical stems pulses clean
