using Newtonsoft.Json;

namespace HRMS.Models.Requests
{

    public class AddMonthlyTransferRequest
    {
        [JsonProperty("TransferId")] public int TransferId { get; set; }

        [JsonProperty("MemberId")] public int MemberId { get; set; }

        [JsonProperty("TransferTypeId")] public int TransferTypeId { get; set; }
        
        [JsonProperty("DayCount")] public int DayCount { get; set; }
        
        [JsonProperty("BeneficiaryId")] public  int BeneficiaryId { get; set; }
        [JsonProperty("Amount")] public double Amount { get; set; }
        
        [JsonProperty("Status")] public int Status { get; set; }
        [JsonProperty("Remark")] public string Remark { get; set; }

        [JsonProperty("TransferDate")] public DateTimeOffset TransferDate { get; set; }
        
        [JsonProperty("PayStartDate")] public DateTimeOffset PayStartDate { get; set; }
        
        [JsonProperty("PayEndDate")] public DateTimeOffset PayEndDate { get; set; }
  
        [JsonIgnore] public int CreateBy { get; set; }

        [JsonIgnore] public int ModifiedBy { get; set; }

    }
}