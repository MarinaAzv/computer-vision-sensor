
import winsound
import time



def alerta_sonoro(frequencia, duracao):
    winsound.Beep(frequencia, duracao)

frequencia = 900  
duracao = 1000  

alerta_sonoro(frequencia, duracao)


