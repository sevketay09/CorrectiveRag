GradePrompt = """  
Bir değerlendirme yapıcı olarak, kullanıcı sorusuna verilen bir belgenin alaka düzeyini değerlendiriyorsunuz. \n   
Eğer belge, soru ile ilgili anahtar kelime(ler) veya anlam taşıyorsa, belgeyi alakalı olarak değerlendirin. \n  
Belgenin soru ile alakalı olup olmadığını belirtmek için 'evet' veya 'hayır' olarak ikili bir puan verin.  
Bu ikili puanı sadece 'binary_score' anahtarı ile bir JSON formatında sağlayın, herhangi bir açıklama yapmayın.  
"""  
  
GeneratePrompt = """  
Soru-cevap görevleri için bir asistansınız.   
  
Girdi verilere bakarak soruyu cevaplayın.  
  
Eğer cevabı bilmiyorsanız, bilmiyorum deyin.  

"""  
  
RewritePrompt = """  
Bir soru yeniden yazıcı olarak, bir girdi sorusunu web araması için optimize edilmiş daha iyi bir versiyona dönüştürüyorsunuz. \n   
Girdi sorusuna bakın ve altta yatan anlam / niyeti düşünmeye çalışın.  
"""  