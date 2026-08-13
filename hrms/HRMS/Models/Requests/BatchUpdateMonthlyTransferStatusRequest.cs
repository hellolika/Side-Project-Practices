using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class BatchUpdateMonthlyTransferStatusRequest
{
    [JsonProperty("StatusId")] public int StatusId { get; set; }
    
    [JsonProperty("TransferIdList")] public List<int> TransferIdList { get; set; }
    
}