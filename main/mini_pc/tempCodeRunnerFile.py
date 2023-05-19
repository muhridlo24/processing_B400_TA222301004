with open('cache/data.json','r+') as file:
#     reader=json.load(file)

#     print(reader[-1])
#     if(len(reader)>0):
#     #threading untuk kirim cache
#         for i in range(len(reader)):
#             print(0,reader[0])
#             resp = requests.post("https://processing-k4ulq4ld5a-et.a.run.app/warning-machine",json=reader[0])
#             print(resp)
#             print(resp.elapsed.total_seconds())
#             """If not then kirim threading"""
#             reader.remove(reader[0])

#     with open('cache/data.json','w') as file:
#         json.dump(reader, file, indent = 4)
#         file.close()