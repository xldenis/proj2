o = Hash[CSV.read('output2').map {|k| [k[0],k[1].strip]}]
ans = lab.select {|k,v| o.keys.include? k }
ans.map {|k,v| o[k] == v}.count(true)