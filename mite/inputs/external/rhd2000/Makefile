#CXX = i686-w64-mingw32-g++
CXX = g++ --std=c++11
CPPFLAGS = -I./rhythm
# -mwin32

LFLAGS = -L./lib -l okFrontPanel -ldl

ifeq ($(OS),Windows_NT)
    CPPFLAGS += -D WIN32
endif

ifeq ($(OS),Windows_NT)
    TARGETNAME = librhd2k.dll
else
	TARGETNAME = librhd2k.so
endif


RDIR = rhythm
ODIR = $(RDIR)/obj

_DEPS = rhd2000evalboard.h rhd2000datablock.h rhd2000registers.h okFrontPanelDLL.h
DEPS = $(patsubst %,$(RDIR)/%,$(_DEPS))
_OBJ = librhd2k.o rhd2000evalboard.o okFrontPanelDLL.o rhd2000datablock.o rhd2000registers.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(RDIR)/%.cpp $(DEPS)
	$(CXX) -c -fPIC -o $@ $< $(CPPFLAGS) $(LFLAGS)

all: directories librhd2k

librhd2k: $(OBJ)
	$(CXX) -shared -o lib/$(TARGETNAME) $^ $(CPPFLAGS) $(LFLAGS)

$(ODIR)/librhd2k.o:
	$(CXX) -c -fPIC rhd2k/python_wrapper.cpp -o $@ $(CPPFLAGS) $(LFLAGS)

directories:
	mkdir -p $(ODIR)

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o
