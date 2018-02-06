function send_email(sender_email, sender_passwd, smtp, ...
  receiver_email, subject, content, attach)
% sender_email
% sender_passwd
% smtp: gmail: smtp.gmail.com
% receiver_email
% subject
% content
% attach
setpref('Internet','E_mail',sender_email);
setpref('Internet','SMTP_Server',smtp);
setpref('Internet','SMTP_Username',sender_email);
setpref('Internet','SMTP_Password',sender_passwd);
props = java.lang.System.getProperties;
props.setProperty('mail.smtp.auth','true');
props.setProperty('mail.smtp.socketFactory.class','javax.net.ssl.SSLSocketFactory');
props.setProperty('mail.smtp.socketFactory.port','465');
try
  if nargin==6
    sendmail(receiver_email,subject,content);
  elseif nargin==7
    sendmail(receiver_email,subject,content,attach);
  end
  disp('Email has been sent successfully!\n');
catch
  disp('Email has NOT been sent!\n');
end
