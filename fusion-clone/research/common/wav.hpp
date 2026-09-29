// Minimal WAV reader/writer (16/24-bit PCM, 32-bit float; mono or multichannel) for the research tools.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace wav {

struct Audio {
	int sampleRate = 48000;
	int channels = 1;
	std::vector<float> data; // interleaved
	size_t frames() const { return channels ? data.size() / channels : 0; }
	std::vector<float> channel(int c) const {
		std::vector<float> o(frames());
		for (size_t i = 0; i < o.size(); i++)
			o[i] = data[i * channels + c];
		return o;
	}
};

inline void w32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
inline void w16(FILE* f, uint16_t v) { fwrite(&v, 2, 1, f); }

/** Write 32-bit float WAV. `interleaved` has frames*channels samples. */
inline bool writeFloat(const std::string& path, const float* interleaved, size_t frames, int channels, int sampleRate) {
	FILE* f = fopen(path.c_str(), "wb");
	if (!f)
		return false;
	uint32_t dataBytes = (uint32_t) (frames * channels * 4);
	fwrite("RIFF", 1, 4, f);
	w32(f, 36 + dataBytes);
	fwrite("WAVE", 1, 4, f);
	fwrite("fmt ", 1, 4, f);
	w32(f, 16);
	w16(f, 3); // IEEE float
	w16(f, (uint16_t) channels);
	w32(f, (uint32_t) sampleRate);
	w32(f, (uint32_t) (sampleRate * channels * 4));
	w16(f, (uint16_t) (channels * 4));
	w16(f, 32);
	fwrite("data", 1, 4, f);
	w32(f, dataBytes);
	fwrite(interleaved, 4, frames * channels, f);
	fclose(f);
	return true;
}

inline bool writeMono(const std::string& path, const std::vector<float>& x, int sampleRate) {
	return writeFloat(path, x.data(), x.size(), 1, sampleRate);
}

inline bool read(const std::string& path, Audio& out) {
	FILE* f = fopen(path.c_str(), "rb");
	if (!f)
		return false;
	char id[4];
	uint32_t sz;
	if (fread(id, 1, 4, f) != 4 || memcmp(id, "RIFF", 4) != 0) {
		fclose(f);
		return false;
	}
	if (fread(&sz, 4, 1, f) != 1 || fread(id, 1, 4, f) != 4) {
		fclose(f);
		return false;
	}
	if (memcmp(id, "WAVE", 4) != 0) {
		fclose(f);
		return false;
	}
	uint16_t fmt = 0, ch = 1, bits = 16;
	uint32_t sr = 48000;
	std::vector<uint8_t> raw;
	bool haveData = false;
	while (fread(id, 1, 4, f) == 4) {
		if (fread(&sz, 4, 1, f) != 1)
			break;
		if (memcmp(id, "fmt ", 4) == 0) {
			std::vector<uint8_t> b(sz);
			if (fread(b.data(), 1, sz, f) != sz || sz < 16) break;
			memcpy(&fmt, &b[0], 2);
			memcpy(&ch, &b[2], 2);
			memcpy(&sr, &b[4], 4);
			memcpy(&bits, &b[14], 2);
			if (fmt == 0xFFFE && sz >= 26)
				memcpy(&fmt, &b[24], 2); // extensible: sub-format tag
		} else if (memcmp(id, "data", 4) == 0) {
			raw.resize(sz);
			size_t got = fread(raw.data(), 1, sz, f);
			raw.resize(got);
			haveData = true;
			break;
		} else {
			fseek(f, (long) sz + (sz & 1), SEEK_CUR);
		}
	}
	fclose(f);
	if (!haveData)
		return false;
	out.sampleRate = (int) sr;
	out.channels = ch;
	size_t bytes = bits / 8;
	size_t n = raw.size() / bytes;
	out.data.resize(n);
	for (size_t i = 0; i < n; i++) {
		const uint8_t* p = &raw[i * bytes];
		float v = 0.f;
		if (fmt == 3 && bits == 32) {
			memcpy(&v, p, 4);
		} else if (bits == 16) {
			int16_t s;
			memcpy(&s, p, 2);
			v = s / 32768.f;
		} else if (bits == 24) {
			int32_t s = (p[0] << 8) | (p[1] << 16) | ((int32_t) p[2] << 24);
			v = (float) (s / 2147483648.0);
		} else if (bits == 32) {
			int32_t s;
			memcpy(&s, p, 4);
			v = (float) (s / 2147483648.0);
		}
		out.data[i] = v;
	}
	return true;
}

} // namespace wav
