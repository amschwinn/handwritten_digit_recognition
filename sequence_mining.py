import pymining

codes = ('146372192346464378', '864748586970878437251433152784', '1111182829393844747383')
#codes = ('2123', '12234', '23345', '34456')
freq_seqs = pymining.freq_seq_enum(codes, 2, 1)
print sorted(freq_seqs, key=lambda x: len(x[0]), reverse=True)


    
    

