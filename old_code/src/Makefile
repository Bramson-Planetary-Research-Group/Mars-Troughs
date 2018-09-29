CC = gcc
ifdef ALONE
ifeq ($(ALONE),yes)
$(info Building executable)
OBJS = driver.o rk4.o equations_of_motion.o
EXEC = test.exe
CFLAGS = -g
OFLAGS = 
endif
else
$(info Building shared library)
OBJS = driver.o rk4.o equations_of_motion.o
EXEC = c_troughs.so
CFLAGS = -fPIC
OFLAGS = -shared
endif

INCL = -I/usr/local/include -O2

LIBS = -lm -L/usr/local/lib -lgsl -lgslcblas

.SUFFIXES:.c.o

%.o: %.c
	$(CC) $(CFLAGS) $(INCL) -c $< -o $@

$(EXEC): $(OBJS)
	$(CC) $(OFLAGS) $(OBJS) $(LIBS) -o $(EXEC)
	rm $(OBJS)

.PHONY: clean

clean:
	rm -f $(OBJS) $(EXEC) *~ *.exe *.so
