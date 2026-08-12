using Newtonsoft.Json;

namespace HRMS.Models.Responses;

public class ResignTransferResponse
{
    [JsonProperty("MemberId")]
    public int MemberId { get; set; }

    [JsonProperty("Name")]
    public string Username { get; set; }
    
    [JsonProperty("Email")]
    public string Email { get; set; }
    
    [JsonProperty("BankAccount")]
    public string BankAccount { get; set; }
    
    [JsonProperty("BankName")]
    public string BankName { get; set; }
    
    [JsonProperty("BaseSalary")]
    public double Salary { get; set; }
    
    [JsonProperty("WorkingDay")]
    public int WorkingDay { get; set; }
    
    [JsonProperty("PaySalary")]
    public double PaySalary { get; set; }
    
    [JsonProperty("UnpaidLeave")]
    public int UnpaidLeave { get; set; }
    
    [JsonProperty("DeductionAmount")]
    public double DeductionAmount { get; set; }
    
    [JsonProperty("NetSalary")]
    public double NetSalary  { get; set; }
}