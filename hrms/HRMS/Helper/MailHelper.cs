using System.Net.Mail;
using HRMS.Services;
using HRMS.Services.Interfaces;

namespace HRMS.Helper;

public class MailHelper
{
    public static void SendMail(string title, string body, string from, string to, ILoggerService loggerService,
        bool? isHtml = false)
    {
        var msgMail = new MailMessage { From = new MailAddress(from) };
        var mailList = to.Split(';');
        foreach (var m in mailList)
        {
            msgMail.To.Add(m);
        }

        msgMail.Subject = title;
        if (isHtml != null) msgMail.IsBodyHtml = (bool)isHtml;
        msgMail.Body = body;
        try
        {
            Smtp.Send(msgMail);
        }
        catch (Exception e)
        {
            loggerService.Error($"Mail Send Fail: Title=>{msgMail.Subject}, To=>{to}");
            throw;
        }
    }

    private static SmtpClient smtp;

    public static SmtpClient Smtp
    {
        get
        {
            if (smtp == null)
            {
                smtp = new SmtpClient("leda-smtp-01.tw01.ppanggu.com");
                smtp.Credentials =
                    new System.Net.NetworkCredential(string.Empty, string.Empty);
                //smtp.EnableSsl = false;
                smtp.UseDefaultCredentials = true;
            }

            return smtp;
        }
    }
}