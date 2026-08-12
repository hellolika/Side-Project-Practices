using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class UpdateMemberMonthlyTransferStatusRequest
{
    [JsonProperty("TransferId")]
    public int TransferId { get; set; }
    
    [JsonProperty("Status")]
    public int Status { get; set; }
}