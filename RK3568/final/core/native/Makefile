CC ?= gcc
PKG_CONFIG ?= pkg-config

CFLAGS ?= -O2 -Wall -Wextra -std=gnu11 -fPIC

DISPLAY_TARGET := librkdrm_display.so
ATOMIC_TARGET := librkdrm_atomic.so
CAPTURE_TARGET := librkhdmi_capture.so
RGA_PREP_TARGET := librga_preprocess.so
RKNN_TARGET := librknn_runtime.so

CAPTURE_PKGS := rockchip_mpp libv4l2
DISPLAY_CFLAGS := $(shell $(PKG_CONFIG) --cflags libdrm librga)
DISPLAY_LIBS := $(shell $(PKG_CONFIG) --libs libdrm librga)
ATOMIC_CFLAGS := $(shell $(PKG_CONFIG) --cflags libdrm freetype2)
ATOMIC_LIBS := $(shell $(PKG_CONFIG) --libs libdrm freetype2) -lpthread
CAPTURE_CPPFLAGS :=
CAPTURE_CFLAGS := $(shell $(PKG_CONFIG) --cflags $(CAPTURE_PKGS))
CAPTURE_LIBS := $(shell $(PKG_CONFIG) --libs $(CAPTURE_PKGS))

RGA_PREP_CFLAGS := $(shell $(PKG_CONFIG) --cflags librga)
RGA_PREP_LIBS := $(shell $(PKG_CONFIG) --libs librga)
RKNN_INCLUDE ?= /home/cat/.local/include
RKNN_CFLAGS := -I$(RKNN_INCLUDE)
RKNN_LIBS := -lrknnrt $(RGA_PREP_LIBS) -lm

.PHONY: all clean

all: $(DISPLAY_TARGET) $(ATOMIC_TARGET) $(CAPTURE_TARGET) $(RGA_PREP_TARGET) $(RKNN_TARGET)

$(DISPLAY_TARGET): rkdrm_display.c
	$(CC) $(CFLAGS) $(DISPLAY_CFLAGS) -shared -o $@ $< $(DISPLAY_LIBS)

$(ATOMIC_TARGET): rkdrm_atomic.c
	$(CC) $(CFLAGS) $(ATOMIC_CFLAGS) -shared -o $@ $< $(ATOMIC_LIBS)

$(CAPTURE_TARGET): rkhdmi_capture.c
	$(CC) $(CFLAGS) $(CAPTURE_CPPFLAGS) $(CAPTURE_CFLAGS) -shared -o $@ $< $(CAPTURE_LIBS)

$(RGA_PREP_TARGET): rga_preprocess.c
	$(CC) $(CFLAGS) $(RGA_PREP_CFLAGS) -shared -o $@ $< $(RGA_PREP_LIBS)

$(RKNN_TARGET): rknn_runtime.c
	$(CC) $(CFLAGS) $(RKNN_CFLAGS) -shared -o $@ $< $(RKNN_LIBS)

clean:
	rm -f $(DISPLAY_TARGET) $(ATOMIC_TARGET) $(CAPTURE_TARGET) $(RGA_PREP_TARGET) $(RKNN_TARGET) *.o
