CREATE PROCEDURE [dbo].[HRMS_EditAnnouncement.1.0.0]
	@id AS INT,
	@title AS NVARCHAR(200),
	@message AS NVARCHAR(2000),
    @modifiedBy AS INT
AS
BEGIN 
	SET NOCOUNT ON
	
	IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[Announcement] WHERE [Id] = @id)
	BEGIN
		SELECT 31 AS ErrorCode;
	END
	ELSE
	BEGIN
		UPDATE [dbo].[Announcement]
		SET [Title] = @title,
		[Message] = @message,
		[ModifiedBy] = @modifiedBy,
		[ModifiedOn] = GETDATE()
		WHERE [Id] = @id;

		SELECT 0 AS ErrorCode;
	END
END