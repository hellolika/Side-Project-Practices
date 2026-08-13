using Newtonsoft.Json;

namespace HRMS.Models;

public class BeneficiaryType
{
    [JsonProperty("TransferId")] public int TransferId { get; set; }
    [JsonProperty("TransferTypeId")] public int TransferTypeId { get; set; }
    
    [JsonProperty("BeneficiaryTypeId")] public int BeneficiaryTypeId { get; set; }
    [JsonProperty("Name")] public string Name { get; set; }
    [JsonProperty("Amount")] public double Amount { get; set; }
    [JsonProperty("DayCount")] public double DayCount { get; set; }
    
    [JsonProperty("TakeLeaveRecord")] public List<TakeLeaveRecord> TakeLeaveRecords { get; set; }
    
    [JsonProperty("Remark")] public string Remark { get; set; }
    
    [JsonProperty("ModifiedBy")] public int ModifiedBy { get; set; }
      
    [JsonProperty("Modifier")] public string Modifier { get; set; }
    
}