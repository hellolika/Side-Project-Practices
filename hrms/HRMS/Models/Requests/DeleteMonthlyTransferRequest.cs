using Newtonsoft.Json;

namespace HRMS.Models.Requests
{

    public class DeleteMonthlyTransferRequest
    {
        [JsonProperty("TransferId")] public int TransferId { get; set; }
    }
}