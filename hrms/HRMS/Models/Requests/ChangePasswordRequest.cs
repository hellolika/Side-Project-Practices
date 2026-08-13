using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class ChangePasswordRequest : PasswordBaseRequest
    {
        [JsonProperty("Password")]
        public string Password { get; set; }

        [JsonProperty("ConfirmPassword")]
        public string ConfirmPassword { get; set; }

        [JsonIgnore]
        public override int MemberId { get; set; }

        public void CheckModels()
        {
            if (string.IsNullOrWhiteSpace(Password))
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Password cannot be empty");
            }
            if (string.IsNullOrWhiteSpace(NewPassword))
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "New Password cannot be empty");
            }
            if (string.IsNullOrWhiteSpace(ConfirmPassword))
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Confirm Password cannot be empty");
            }
            if (!NewPassword.Equals(ConfirmPassword, StringComparison.InvariantCulture))
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "New password and confirm password do not match");
            }
        }
    }
}