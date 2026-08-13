SOURCE := monografia.tex
TECHNICAL_PDF := monografia.pdf
FINAL_PDF := 2026_GiovanniMinariZanetti_MateusGomesDeAraujo_tcc.pdf

.PHONY: all final clean

all: final

final:
	latexmk -pdf $(SOURCE)
	mv $(TECHNICAL_PDF) $(FINAL_PDF)
	@echo "PDF final atualizado: $(FINAL_PDF)"

clean:
	latexmk -c $(SOURCE)
	rm -f $(TECHNICAL_PDF)
