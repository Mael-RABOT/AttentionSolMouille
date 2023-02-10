SRC	=	main.py
EXE	=	run.py

all:	$(EXE)

$(EXE):
		cp $(SRC) $(EXE)
		chmod 755 $(EXE)

clean:
		rm -rf $(EXE)

re:	clean
	all