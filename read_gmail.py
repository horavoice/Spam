import sys
import gmail
g = gmail.login(str(sys.argv[1]), str(sys.argv[2]))
if len(sys.argv) > 3 and str(sys.argv[3]) == 'spam':
	emails = g.mailbox("[Gmail]/Spam").mail(prefetch=True)
else:
	emails = g.inbox().mail(prefetch=True)
for index, email in enumerate(emails):
    with open(str(index) + ".txt", "w") as file:
        file.write(email.body)
g.logout()